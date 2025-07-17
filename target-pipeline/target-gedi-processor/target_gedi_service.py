import os
import sys
import json
import time
import zipfile
import glob
import numpy as np
import open3d as o3d
import torch
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

# Add GeDi and PointNet2 to Python path
sys.path.insert(0, '/app/gedi-repo')
sys.path.insert(0, '/app/pointnet2')

try:
    from gedi import GeDi
    print("✅ GeDi imported successfully")
except ImportError as e:
    print(f"❌ Failed to import GeDi: {e}")
    sys.exit(1)

class TargetGeDiProcessor:
    """Target GeDi processor for geometric feature extraction from point clouds"""
    
    def __init__(self):
        self.system_resources = self.detect_system_resources()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        
        # Initialize GeDi models with different radii
        self.setup_gedi_models()
        
        print("🔷 Target GeDi Processor Service Ready")
    
    def detect_system_resources(self):
        """Detect and optimize system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        resources = {
            'cpu_count': cpu_count,
            'memory_total': memory_total,
            'cuda_available': cuda_available,
            'gpu_count': gpu_count
        }
        
        print(f"🚀 System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 CUDA Available: {cuda_available}")
        print(f"   🔧 GPU Count: {gpu_count}")
        
        return resources
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                'target-gedi-processing',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='target-gedi-processor-group',
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
            thread_name_prefix="gedi_worker"
        )
        
        print("✅ Thread executors initialized")
    
    def setup_gedi_models(self):
        """Initialize GeDi models with different radii configurations"""
        try:
            # Base configuration for GeDi models
            base_config = {
                'dim': 32,
                'samples_per_batch': 500,
                'samples_per_patch_lrf': 4000,
                'samples_per_patch_out': 512,
                'fchkpt_gedi_net': '/app/gedi-repo/data/chkpts/3dmatch/chkpt.tar'
            }
            
            print("✅ GeDi models configuration ready")
            self.base_config = base_config
            
        except Exception as e:
            print(f"❌ Failed to setup GeDi models: {e}")
            raise
    
    def normalize_pcd_max_extent(self, pcd):
        """Normalize point cloud to unit scale based on maximum extent"""
        aabb = pcd.get_axis_aligned_bounding_box()
        extents = np.asarray(aabb.get_extent())
        max_extent = extents.max()
        
        if max_extent == 0:
            raise ValueError("Point cloud has zero size along all axes.")
        
        pcd.scale(1.0 / max_extent, center=(0, 0, 0))
        return pcd
    
    def natural_sort_key(self, path):
        """Natural sorting for frame ordering"""
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', path)]
    
    def ensure_numpy(self, x):
        """Convert tensor or array to numpy array"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        raise TypeError(f"Unexpected type: {type(x)} — expected Tensor or ndarray.")
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from PLY filename"""
        # Handle target_XXXXXX.ply or pointcloud_XXXXXX.ply formats
        name = Path(filename).stem
        
        patterns = [
            r'target_(\d+)',
            r'pointcloud_(\d+)',
            r'(\d{6})',
            r'(\d{5})',
            r'(\d{4})',
            r'(\d{3})',
            r'(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def process_single_point_cloud(self, ply_path: str, output_dir: Path) -> str:
        """Process a single point cloud to extract GeDi features"""
        
        try:
            filename = os.path.basename(ply_path)
            frame_id = self.extract_frame_id(filename)
            
            print(f"🔷 Processing: {frame_id}")
            
            # Load and normalize point cloud
            pcd = o3d.io.read_point_cloud(ply_path)
            
            if len(pcd.points) == 0:
                raise ValueError(f"Empty point cloud: {ply_path}")
            
            pcd = self.normalize_pcd_max_extent(pcd)
            
            # Estimate diameter for radius calculation
            aabb = pcd.get_axis_aligned_bounding_box()
            min_b, max_b = np.asarray(aabb.min_bound), np.asarray(aabb.max_bound)
            diameter = np.linalg.norm(max_b - min_b)
            
            # Convert to tensor
            pts = torch.tensor(np.asarray(pcd.points)).float()
            
            # Initialize GeDi models with frame-specific radius
            config_03 = self.base_config.copy()
            config_03['r_lrf'] = 0.3 * diameter
            
            config_04 = self.base_config.copy()
            config_04['r_lrf'] = 0.4 * diameter
            
            gedi_03 = GeDi(config=config_03)
            gedi_04 = GeDi(config=config_04)
            
            # Compute descriptors for both radii
            desc_03 = gedi_03.compute(pts=pts, pcd=pts)
            desc_04 = gedi_04.compute(pts=pts, pcd=pts)
            
            # Ensure outputs are numpy arrays
            desc_03_np = self.ensure_numpy(desc_03)
            desc_04_np = self.ensure_numpy(desc_04)
            
            # Concatenate features → N x 64
            desc_combined = np.concatenate([desc_03_np, desc_04_np], axis=1)
            
            # Verify feature dimensions
            if desc_combined.shape[1] != 64:
                raise ValueError(f"Expected 64 features per point, got {desc_combined.shape[1]}")
            
            # Save numpy array with exact naming convention
            save_name = f"target_{frame_id}_gedi.npy"
            save_path = output_dir / save_name
            np.save(save_path, desc_combined)
            
            print(f"✅ Saved {save_name} with shape {desc_combined.shape}")
            
            # Clean up GPU memory
            del gedi_03, gedi_04, desc_03, desc_04, pts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return save_name
            
        except Exception as e:
            print(f"❌ Failed to process {ply_path}: {e}")
            return None
    
    def process_target_gedi_sequence(self, message: Dict[str, Any]):
        """Process target sequence for GeDi geometric features"""
        
        try:
            job_id = message['job_id']
            sequence_name = message['sequence_name']
            point_clouds_zip = message['point_clouds_zip']
            
            print(f"🔷 Processing GeDi features for: {sequence_name}")
            print(f"📋 Job ID: {job_id}")
            print(f"📁 Point clouds ZIP: {point_clouds_zip}")
            
            # Create output directory
            output_dir = Path(f"/app/output/{job_id}/gedi_features")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                extract_dir = temp_path / "point_clouds"
                extract_dir.mkdir()
                
                # Extract point clouds ZIP
                print("📦 Extracting point clouds ZIP...")
                with zipfile.ZipFile(point_clouds_zip, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find all PLY files
                ply_files = glob.glob(str(extract_dir / "*.ply"))
                ply_files.sort(key=self.natural_sort_key)
                
                num_frames = len(ply_files)
                print(f"📊 Found {num_frames} point cloud files")
                
                if num_frames == 0:
                    raise ValueError("No PLY files found in point clouds ZIP")
                
                # Process point clouds in parallel
                output_files = []
                
                def process_worker(ply_path):
                    return self.process_single_point_cloud(ply_path, output_dir)
                
                print("🔷 Extracting GeDi features with dual-scale processing...")
                
                # Use thread pool for parallel processing
                max_workers = min(self.system_resources['cpu_count'], num_frames, 4)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ply = {
                        executor.submit(process_worker, ply_path): ply_path
                        for ply_path in ply_files
                    }
                    
                    for future in tqdm(as_completed(future_to_ply), 
                                     total=len(ply_files), 
                                     desc="Processing PLY files"):
                        result = future.result()
                        if result:
                            output_files.append(result)
                
                # Create ZIP file of geometric features
                geometric_features_zip = output_dir.parent / "geometric_features.zip"
                
                print("📦 Creating geometric features ZIP...")
                with zipfile.ZipFile(geometric_features_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for feature_file in output_files:
                        feature_path = output_dir / feature_file
                        if feature_path.exists():
                            zip_ref.write(feature_path, feature_file)
                
                print(f"✅ GeDi processing completed!")
                print(f"📊 Processed {len(output_files)} point clouds")
                print(f"💾 Geometric features saved to: {geometric_features_zip}")
                
                # Send completion message
                completion_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'geometric_features_zip': str(geometric_features_zip),
                    'total_frames': len(output_files),
                    'processing_type': 'target_gedi_complete',
                    'feature_dimensions': 64,
                    'timestamp': time.time()
                }
                
                self.producer.send('target-feature-fusion', completion_message)
                
                print(f"✅ GeDi processing completed for {job_id}")
                
                # Cleanup GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Target GeDi processing failed: {e}")
            # Send error message
            error_message = {
                'job_id': message.get('job_id', 'unknown'),
                'sequence_name': message.get('sequence_name', 'unknown'),
                'error': str(e),
                'processing_type': 'target_gedi_error',
                'timestamp': time.time()
            }
            self.producer.send('target-processing-error', error_message)
    
    def run(self):
        """Run the target GeDi processor service"""
        print("🚀 Starting Target GeDi Processor Service")
        print(f"💻 CPU Cores: {self.system_resources['cpu_count']}")
        print(f"📊 Memory: {self.system_resources['memory_total']:.1f}GB")
        print(f"🎮 CUDA Available: {self.system_resources['cuda_available']}")
        print("🔄 Waiting for point cloud processing requests...")
        
        try:
            for message in self.consumer:
                self.process_target_gedi_sequence(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Target GeDi Processor")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    processor = TargetGeDiProcessor()
    processor.run()
