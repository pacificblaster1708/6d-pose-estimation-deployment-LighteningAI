import os
import json
import time
import zipfile
import numpy as np
import cv2
import open3d as o3d
from typing import Dict, Any, List
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from tqdm import tqdm
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import re

class PointCloudGenerator:
    """High-performance point cloud generator with dynamic camera intrinsics"""
    
    def __init__(self):
        self.system_resources = self.detect_system_resources()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        
        print("🔷 Point Cloud Generator Service Ready")
    
    def detect_system_resources(self):
        """Detect and optimize system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        resources = {
            'cpu_count': cpu_count,
            'memory_total': memory_total
        }
        
        print(f"🚀 System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        
        return resources
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                'target-point-cloud-generation',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='point-cloud-generator-group',
                fetch_max_wait_ms=100
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=16384,
                linger_ms=10
            )
            print("✅ Kafka connected")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
    
    def setup_executors(self):
        """Setup optimized thread pools"""
        self.io_executor = ThreadPoolExecutor(
            max_workers=min(self.system_resources['cpu_count'], 16),
            thread_name_prefix="pcg_io_worker"
        )
        
        print("✅ High-performance executors initialized")
    
    def load_camera_intrinsics_dynamic(self, intrinsics_path: str, frame_id: str) -> Dict[str, float]:
        """Dynamically load camera intrinsics from JSON file"""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Try multiple possible keys for frame-specific intrinsics
            possible_keys = [
                frame_id,
                f"frame_{frame_id}",
                f"{frame_id}.jpg",
                f"{frame_id}.png",
                "default"
            ]
            
            frame_intrinsics = None
            for key in possible_keys:
                if key in intrinsics_data:
                    frame_intrinsics = intrinsics_data[key]
                    break
            
            if frame_intrinsics is None:
                if isinstance(intrinsics_data, dict) and intrinsics_data:
                    frame_intrinsics = list(intrinsics_data.values())[0]
                else:
                    raise ValueError("No camera intrinsics found")
            
            # Handle different intrinsics formats
            if isinstance(frame_intrinsics, dict):
                intrinsics = {
                    'fx': float(frame_intrinsics.get('fx', 525.0)),
                    'fy': float(frame_intrinsics.get('fy', 525.0)),
                    'cx': float(frame_intrinsics.get('cx', 320.0)),
                    'cy': float(frame_intrinsics.get('cy', 240.0))
                }
            elif isinstance(frame_intrinsics, list) and len(frame_intrinsics) >= 9:
                intrinsics = {
                    'fx': float(frame_intrinsics[0]),
                    'fy': float(frame_intrinsics[4]),
                    'cx': float(frame_intrinsics[2]),
                    'cy': float(frame_intrinsics[5])
                }
            elif isinstance(frame_intrinsics, list) and len(frame_intrinsics) == 4:
                intrinsics = {
                    'fx': float(frame_intrinsics[0]),
                    'fy': float(frame_intrinsics[1]),
                    'cx': float(frame_intrinsics[2]),
                    'cy': float(frame_intrinsics[3])
                }
            else:
                raise ValueError(f"Unsupported intrinsics format")
            
            return intrinsics
            
        except Exception as e:
            print(f"❌ Failed to load intrinsics for frame {frame_id}: {e}")
            return {'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0}
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from filename"""
        name = Path(filename).stem
        patterns = [r'(\d{6})', r'(\d{5})', r'(\d{4})', r'(\d{3})', r'(\d+)']
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def find_corresponding_file(self, frame_id: str, file_list: List[str]) -> str:
        """Find corresponding file based on frame ID"""
        for filename in file_list:
            if frame_id in filename:
                return filename
        
        # Try without leading zeros
        frame_id_no_zeros = frame_id.lstrip('0') or '0'
        for filename in file_list:
            if frame_id_no_zeros in filename:
                return filename
        
        return None
    
    def generate_point_cloud_from_rgbd(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                                     mask: np.ndarray, intrinsics: Dict[str, float], 
                                     frame_id: str) -> o3d.geometry.PointCloud:
        """Generate point cloud from RGB-D data with mask"""
        
        try:
            # Resize RGB to match depth dimensions
            if rgb_image.shape[:2] != depth_image.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))
            
            # Resize mask if needed
            if mask.shape != depth_image.shape:
                mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Apply mask
            mask_binary = (mask > 128).astype(np.uint8)
            masked_depth = depth_image.copy()
            masked_depth[mask_binary == 0] = 0
            masked_rgb = rgb_image.copy()
            masked_rgb[mask_binary == 0] = 0
            
            # Create Open3D images
            color_o3d = o3d.geometry.Image(cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=color_o3d,
                depth=depth_o3d,
                convert_rgb_to_intensity=False,
                depth_scale=1000.0,
                depth_trunc=30.0
            )
            
            # Create camera intrinsic parameters
            intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
                width=depth_image.shape[1],
                height=depth_image.shape[0],
                fx=intrinsics['fx'],
                fy=intrinsics['fy'],
                cx=intrinsics['cx'],
                cy=intrinsics['cy']
            )
            
            # Generate point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix)
            
            # Remove outliers for cleaner point clouds
            if len(point_cloud.points) > 50:
                point_cloud, _ = point_cloud.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
            
            # Downsample to 1000 points
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors)
            
            if len(points) > 1000:
                idx = np.random.choice(len(points), 1000, replace=False)
                points_sampled = points[idx]
                colors_sampled = colors[idx]
            else:
                points_sampled = points
                colors_sampled = colors
                print(f"⚠️ Frame {frame_id}: Only {len(points)} points found (target: 1000)")
            
            # Create final point cloud
            pcd_sampled = o3d.geometry.PointCloud()
            pcd_sampled.points = o3d.utility.Vector3dVector(points_sampled)
            pcd_sampled.colors = o3d.utility.Vector3dVector(colors_sampled)
            
            return pcd_sampled
            
        except Exception as e:
            print(f"❌ Point cloud generation failed for frame {frame_id}: {e}")
            return o3d.geometry.PointCloud()
    
    def process_single_frame(self, frame_data: Dict[str, Any], temp_dirs: Dict[str, Path], 
                           intrinsics_path: str, output_dir: Path) -> str:
        """Process a single frame to generate point cloud"""
        
        try:
            frame_id = frame_data['frame_id']
            
            # Load dynamic camera intrinsics
            intrinsics = self.load_camera_intrinsics_dynamic(intrinsics_path, frame_id)
            
            # Load images
            rgb_path = temp_dirs['rgb'] / frame_data['rgb_frame']
            rgb_image = cv2.imread(str(rgb_path))
            
            depth_path = temp_dirs['depth'] / frame_data['depth_frame']
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            
            mask_path = temp_dirs['masks'] / frame_data['mask_frame']
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if rgb_image is None or depth_image is None or mask is None:
                raise ValueError(f"Failed to load images for frame {frame_id}")
            
            # Generate point cloud
            point_cloud = self.generate_point_cloud_from_rgbd(
                rgb_image, depth_image, mask, intrinsics, frame_id
            )
            
            # Save point cloud
            pc_filename = f"pointcloud_{frame_id}.ply"
            pc_path = output_dir / pc_filename
            
            success = o3d.io.write_point_cloud(str(pc_path), point_cloud)
            
            if success:
                print(f"✅ Generated point cloud for frame {frame_id}: {len(point_cloud.points)} points")
                return pc_filename
            else:
                print(f"❌ Failed to save point cloud for frame {frame_id}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to process frame {frame_data.get('frame_id', 'unknown')}: {e}")
            return None
    
    def process_target_sequence(self, message: Dict[str, Any]):
        """Process target video sequence to generate point clouds"""
        
        try:
            job_id = message['job_id']
            sequence_name = message['sequence_name']
            file_paths = message['file_paths']
            frame_mapping = message['frame_mapping']
            
            print(f"🎬 Processing target sequence: {sequence_name}")
            print(f"📋 Total frames: {len(frame_mapping)}")
            
            # Create output directory
            output_dir = Path(f"/app/output/{job_id}/point_clouds")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary extraction directories
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                temp_dirs = {
                    'rgb': temp_path / "rgb",
                    'masks': temp_path / "masks", 
                    'depth': temp_path / "depth"
                }
                
                for dir_path in temp_dirs.values():
                    dir_path.mkdir()
                
                print("📦 Extracting ZIP files in parallel...")
                
                def extract_zip(args):
                    zip_path, extract_dir = args
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                
                extraction_tasks = [
                    (file_paths['rgb_frames'], temp_dirs['rgb']),
                    (file_paths['binary_masks'], temp_dirs['masks']),
                    (file_paths['depth_images'], temp_dirs['depth'])
                ]
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    list(executor.map(extract_zip, extraction_tasks))
                
                print("✅ ZIP extraction completed")
                
                # Process frames in parallel
                point_cloud_files = []
                
                print("🔷 Generating point clouds...")
                
                max_workers = min(self.system_resources['cpu_count'], len(frame_mapping), 8)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_frame = {
                        executor.submit(
                            self.process_single_frame,
                            frame_data, 
                            temp_dirs, 
                            file_paths['camera_intrinsics'], 
                            output_dir
                        ): frame_data
                        for frame_data in frame_mapping
                    }
                    
                    for future in tqdm(as_completed(future_to_frame), 
                                     total=len(frame_mapping), 
                                     desc="Processing frames"):
                        result = future.result()
                        if result:
                            point_cloud_files.append(result)
                
                # Create ZIP file of point clouds
                point_clouds_zip = output_dir.parent / "point_clouds.zip"
                with zipfile.ZipFile(point_clouds_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for pc_file in point_cloud_files:
                        pc_path = output_dir / pc_file
                        if pc_path.exists():
                            zip_ref.write(pc_path, pc_file)
                
                print(f"✅ Generated {len(point_cloud_files)} point clouds")
                print(f"💾 Point clouds ZIP: {point_clouds_zip}")
                
                # Send completion message for parallel DINOv2 and GeDi processing
                completion_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'point_clouds_zip': str(point_clouds_zip),
                    'total_frames': len(point_cloud_files),
                    'file_paths': file_paths,
                    'frame_mapping': frame_mapping,
                    'processing_type': 'point_cloud_generation_complete',
                    'timestamp': time.time()
                }
                
                # Send to both DINOv2 and GeDi processors in parallel
                self.producer.send('target-dinov2-processing', completion_message)
                self.producer.send('target-gedi-processing', completion_message)
                
                print(f"✅ Point cloud generation completed for {job_id}")
                
                gc.collect()
                
        except Exception as e:
            print(f"❌ Target sequence processing failed: {e}")
    
    def run(self):
        """Run the point cloud generator service"""
        print("🚀 Starting Point Cloud Generator Service")
        print(f"💻 CPU Cores: {self.system_resources['cpu_count']}")
        print(f"📊 Memory: {self.system_resources['memory_total']:.1f}GB")
        print("🔄 Waiting for target sequence processing requests...")
        
        try:
            for message in self.consumer:
                self.process_target_sequence(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Point Cloud Generator")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    generator = PointCloudGenerator()
    generator.run()
