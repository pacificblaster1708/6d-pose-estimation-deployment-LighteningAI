import os
import json
import time
import zipfile
import numpy as np
import torch
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import tempfile
import gc

# Computer vision and 3D processing
import cv2
from PIL import Image
import open3d as o3d

# Machine learning
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Kafka integration
from kafka import KafkaConsumer, KafkaProducer

# Performance monitoring
import psutil
from tqdm import tqdm
import re

class GPUAcceleratedDINOv2Processor:
    """GPU-accelerated DINOv2 processor for target video sequences"""
    
    def __init__(self):
        self.system_resources = self.detect_system_resources()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        self.setup_dinov2()
        
        print("🎨 GPU-Accelerated Target DINOv2 Processor Service Ready")
    
    def detect_system_resources(self):
        """Detect and optimize system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        resources = {
            'cpu_count': cpu_count,
            'memory_total': memory_total,
            'gpu_available': gpu_available,
            'gpu_count': gpu_count
        }
        
        print(f"🚀 System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Available: {gpu_available}")
        print(f"   🔧 GPU Count: {gpu_count}")
        
        return resources
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                'target-dinov2-processing',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='target-dinov2-processor-group',
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
            max_workers=min(self.system_resources['cpu_count'], 8),
            thread_name_prefix="dinov2_io"
        )
        print("✅ Thread pools initialized")
    
    def setup_dinov2(self):
        """Setup DINOv2 model with GPU optimization"""
        print("🔄 Loading DINOv2 Giant model...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        self.model = AutoModel.from_pretrained("facebook/dinov2-giant").to(self.device).eval()
        
        # Optimize for inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"✅ DINOv2 Giant loaded on {self.device}")
        print(f"   📐 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_camera_intrinsics_dynamic(self, intrinsics_path: str, frame_id: str) -> np.ndarray:
        """Load camera intrinsics from JSON file for specific frame"""
        try:
            with open(intrinsics_path, 'r') as f:
                intrinsics_data = json.load(f)
            
            # Try multiple keys for frame-specific intrinsics
            frame_id_no_zeros = frame_id.lstrip('0') or '0'
            possible_keys = [
                frame_id_no_zeros,
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
            
            # Handle different intrinsics formats
            if isinstance(frame_intrinsics, dict):
                if 'cam_K' in frame_intrinsics:
                    # Scene camera format
                    cam_K = frame_intrinsics['cam_K']
                    return np.array(cam_K).reshape(3, 3)
                else:
                    # Direct format
                    fx = float(frame_intrinsics.get('fx', 525.0))
                    fy = float(frame_intrinsics.get('fy', 525.0))
                    cx = float(frame_intrinsics.get('cx', 320.0))
                    cy = float(frame_intrinsics.get('cy', 240.0))
                    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            elif isinstance(frame_intrinsics, list) and len(frame_intrinsics) >= 9:
                # Matrix format
                return np.array(frame_intrinsics).reshape(3, 3)
            
            # Fallback default intrinsics
            return np.array([
                [1066.778, 0.0, 312.9869],
                [0.0, 1067.487, 241.3109],
                [0.0, 0.0, 1.0]
            ])
            
        except Exception as e:
            print(f"❌ Failed to load intrinsics for frame {frame_id}: {e}")
            # Return default intrinsics
            return np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
    
    def project_points_to_image(self, points: np.ndarray, intrinsic: np.ndarray) -> tuple:
        """Project 3D points to image coordinates"""
        z = points[:, 2]
        u = (points[:, 0] * intrinsic[0, 0]) / z + intrinsic[0, 2]
        v = (points[:, 1] * intrinsic[1, 1]) / z + intrinsic[1, 2]
        uv = np.stack([u, v], axis=1)
        valid = z > 0
        return uv, valid
    
    def get_patch_indices(self, uv: np.ndarray, image_shape: tuple, patch_size: int = 14) -> np.ndarray:
        """Get patch indices for DINOv2 feature mapping"""
        h_patches = image_shape[0] // patch_size
        w_patches = image_shape[1] // patch_size
        
        u_idx = np.clip((uv[:, 0] / patch_size).astype(int), 0, w_patches - 1)
        v_idx = np.clip((uv[:, 1] / patch_size).astype(int), 0, h_patches - 1)
        
        return v_idx * w_patches + u_idx
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from filename"""
        name = Path(filename).stem
        
        # Handle various naming conventions
        patterns = [r'(\d{6})', r'(\d{5})', r'(\d{4})', r'(\d{3})', r'(\d+)']
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def extract_dino_features_with_pca(self, rgb_image: Image.Image, point_cloud: o3d.geometry.PointCloud, 
                                     intrinsic: np.ndarray, frame_id: str) -> np.ndarray:
        """Extract DINOv2 features and apply PCA reduction with L2 normalization"""
        
        try:
            # Get image dimensions
            image_size = rgb_image.size[::-1]  # (height, width)
            patch_size = 14
            
            # Get point cloud points
            points = np.asarray(point_cloud.points)
            assert points.shape[0] == 1000, f"Expected 1000 points, got {points.shape[0]}"
            
            print(f"📸 Extracting DINOv2 features for frame {frame_id}")
            
            # Process image through DINOv2
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model(**inputs)
            
            # Get feature tokens (exclude CLS token)
            dino_features = output.last_hidden_state[0, 1:, :].cpu().numpy()  # [tokens, 1536]
            
            print(f"   🎯 DINOv2 features shape: {dino_features.shape}")
            
            # Project points to image
            uv, valid = self.project_points_to_image(points, intrinsic)
            
            # Get patch indices for valid points
            patch_indices = self.get_patch_indices(uv[valid], image_size, patch_size)
            patch_indices = np.clip(patch_indices, 0, dino_features.shape[0] - 1)
            
            # Initialize feature array for all points
            point_features = np.zeros((points.shape[0], dino_features.shape[1]), dtype=np.float32)
            
            # Assign features to valid points
            point_features[valid] = dino_features[patch_indices]
            
            print(f"   📊 Point features shape: {point_features.shape}")
            print(f"   ✅ Valid points: {valid.sum()}/{len(valid)}")
            
            # Apply PCA to reduce to 64D
            print("🎛️ Reducing with PCA → 64D")
            pca = PCA(n_components=64)
            reduced_features = pca.fit_transform(point_features)
            
            # Apply L2 normalization
            print("🔄 Applying L2 normalization")
            normalized_features = normalize(reduced_features, norm='l2', axis=1)
            
            print(f"   🎯 Final features shape: {normalized_features.shape}")
            print(f"   📏 Feature norm check: {np.linalg.norm(normalized_features[0]):.4f} (should be ~1.0)")
            
            return normalized_features.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Feature extraction failed for frame {frame_id}: {e}")
            # Return zero features as fallback
            return np.zeros((1000, 64), dtype=np.float32)
    
    def process_single_frame(self, frame_data: Dict[str, Any], temp_dirs: Dict[str, Path], 
                           intrinsics_path: str, output_dir: Path) -> str:
        """Process a single frame for DINOv2 feature extraction"""
        
        try:
            frame_id = frame_data['frame_id']
            
            print(f"🎨 Processing DINOv2 features for frame {frame_id}")
            
            # Load RGB image
            rgb_path = temp_dirs['rgb'] / frame_data['rgb_frame']
            rgb_image = Image.open(str(rgb_path)).convert("RGB")
            
            # Load point cloud
            pc_path = temp_dirs['point_clouds'] / frame_data['pointcloud_file']
            point_cloud = o3d.io.read_point_cloud(str(pc_path))
            
            # Load camera intrinsics
            intrinsic = self.load_camera_intrinsics_dynamic(intrinsics_path, frame_id)
            
            # Apply mask if available
            if 'mask_frame' in frame_data and frame_data['mask_frame']:
                try:
                    mask_path = temp_dirs.get('masks', Path()) / frame_data['mask_frame']
                    if mask_path.exists():
                        print(f"🎭 Applying mask for frame {frame_id}")
                        mask = Image.open(str(mask_path)).convert("L")
                        rgb_np = np.array(rgb_image)
                        mask_np = np.array(mask)
                        rgb_np[mask_np == 0] = 0
                        rgb_image = Image.fromarray(rgb_np)
                except Exception as e:
                    print(f"⚠️ Mask application failed for frame {frame_id}: {e}")
            
            # Extract DINOv2 features with PCA and L2 normalization
            features = self.extract_dino_features_with_pca(
                rgb_image, point_cloud, intrinsic, frame_id
            )
            
            # Save features
            feature_filename = f"dinov2_features_{frame_id}.npy"
            feature_path = output_dir / feature_filename
            np.save(str(feature_path), features)
            
            print(f"✅ Saved DINOv2 features for frame {frame_id}: {features.shape}")
            return feature_filename
            
        except Exception as e:
            print(f"❌ Failed to process DINOv2 features for frame {frame_data.get('frame_id', 'unknown')}: {e}")
            return None
    
    def process_target_dinov2_sequence(self, message: Dict[str, Any]):
        """Process target sequence for DINOv2 visual feature extraction"""
        
        try:
            job_id = message['job_id']
            sequence_name = message['sequence_name']
            point_clouds_zip = message['point_clouds_zip']
            file_paths = message['file_paths']
            frame_mapping = message['frame_mapping']
            
            print(f"🎨 Processing DINOv2 features for sequence: {sequence_name}")
            print(f"📋 Total frames: {len(frame_mapping)}")
            
            # Create output directory
            output_dir = Path(f"/app/output/{job_id}/dinov2_features")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary extraction directories
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                temp_dirs = {
                    'rgb': temp_path / "rgb",
                    'point_clouds': temp_path / "point_clouds"
                }
                
                # Add masks directory if available
                if 'binary_masks' in file_paths:
                    temp_dirs['masks'] = temp_path / "masks"
                
                for dir_path in temp_dirs.values():
                    dir_path.mkdir()
                
                print("📦 Extracting files in parallel...")
                
                # Extract RGB frames
                with zipfile.ZipFile(file_paths['rgb_frames'], 'r') as zip_ref:
                    zip_ref.extractall(temp_dirs['rgb'])
                
                # Extract point clouds
                with zipfile.ZipFile(point_clouds_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_dirs['point_clouds'])
                
                # Extract masks if available
                if 'masks' in temp_dirs and 'binary_masks' in file_paths:
                    with zipfile.ZipFile(file_paths['binary_masks'], 'r') as zip_ref:
                        zip_ref.extractall(temp_dirs['masks'])
                
                print("✅ File extraction completed")
                
                # Create frame mapping with point cloud files
                enhanced_frame_mapping = []
                for frame_data in frame_mapping:
                    frame_id = frame_data['frame_id']
                    pointcloud_file = f"pointcloud_{frame_id}.ply"
                    
                    if (temp_dirs['point_clouds'] / pointcloud_file).exists():
                        enhanced_frame_mapping.append({
                            **frame_data,
                            'pointcloud_file': pointcloud_file
                        })
                    else:
                        print(f"⚠️ Point cloud not found for frame {frame_id}")
                
                # Process frames in parallel
                feature_files = []
                
                print("🎨 Extracting DINOv2 features with GPU acceleration...")
                
                max_workers = min(2, len(enhanced_frame_mapping))  # Limit for GPU memory
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_frame = {
                        executor.submit(
                            self.process_single_frame,
                            frame_data,
                            temp_dirs,
                            file_paths['camera_intrinsics'],
                            output_dir
                        ): frame_data
                        for frame_data in enhanced_frame_mapping
                    }
                    
                    for future in tqdm(as_completed(future_to_frame), 
                                     total=len(enhanced_frame_mapping), 
                                     desc="DINOv2 features"):
                        result = future.result()
                        if result:
                            feature_files.append(result)
                
                # Create ZIP file of visual features
                visual_features_zip = output_dir.parent / "visual_features.zip"
                with zipfile.ZipFile(visual_features_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for feature_file in feature_files:
                        feature_path = output_dir / feature_file
                        if feature_path.exists():
                            zip_ref.write(feature_path, feature_file)
                
                print(f"✅ Generated {len(feature_files)} DINOv2 feature files")
                print(f"💾 Visual features ZIP: {visual_features_zip}")
                
                # Send completion message
                completion_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'visual_features_zip': str(visual_features_zip),
                    'total_frames': len(feature_files),
                    'processing_type': 'target_dinov2_complete',
                    'timestamp': time.time()
                }
                
                self.producer.send('target-feature-fusion', completion_message)
                
                print(f"✅ DINOv2 processing completed for {job_id}")
                
                # Cleanup GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Target DINOv2 sequence processing failed: {e}")
    
    def run(self):
        """Run the target DINOv2 processor service"""
        print("🚀 Starting GPU-Accelerated Target DINOv2 Processor Service")
        print(f"💻 CPU Cores: {self.system_resources['cpu_count']}")
        print(f"📊 Memory: {self.system_resources['memory_total']:.1f}GB")
        print(f"🎮 GPU Available: {self.system_resources['gpu_available']}")
        print("🔄 Waiting for DINOv2 processing requests...")
        
        try:
            for message in self.consumer:
                self.process_target_dinov2_sequence(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Target DINOv2 Processor")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    processor = GPUAcceleratedDINOv2Processor()
    processor.run()
