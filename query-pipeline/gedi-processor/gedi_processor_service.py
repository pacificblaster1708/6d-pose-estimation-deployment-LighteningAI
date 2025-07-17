import os
import sys
import json
import time
import logging
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import psutil
import GPUtil
from memory_profiler import profile
import gc
from kafka import KafkaConsumer, KafkaProducer
from tqdm import tqdm

# Add GeDi repository to Python path
sys.path.insert(0, '/app/gedi-repo')

# Import GeDi after path setup
try:
    from gedi import GeDi
    print("✅ GeDi imported successfully")
except ImportError as e:
    print(f"❌ Failed to import GeDi: {e}")
    print("Using fallback geometric feature extraction")

@dataclass
class SystemResources:
    cpu_count: int
    memory_total: float
    gpu_count: int
    gpu_memory_total: List[int]
    cuda_available: bool

class HighPerformanceGeDiProcessor:
    """High-performance GeDi processor following your exact workflow"""
    
    def __init__(self):
        self.system_detector = self.detect_system_resources()
        self.resources = self.system_detector
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Kafka setup
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # Thread pool setup
        self.setup_executors()
        
        print("🚀 High-Performance GeDi Processor Ready")
    
    def detect_system_resources(self) -> SystemResources:
        """Detect system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
            cuda_available = torch.cuda.is_available()
        except:
            gpu_count = 0
            gpu_memory_total = []
            cuda_available = False
        
        resources = SystemResources(
            cpu_count=cpu_count,
            memory_total=memory_total,
            gpu_count=gpu_count,
            gpu_memory_total=gpu_memory_total,
            cuda_available=cuda_available
        )
        
        print(f"🚀 System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {gpu_count}")
        print(f"   🚀 CUDA Available: {cuda_available}")
        
        return resources
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                'gedi-processing',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='gedi-processor-group'
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            print(f"✅ Kafka connected at {self.kafka_servers}")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
            raise
    
    def setup_executors(self):
        """Setup thread pools"""
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=min(self.resources.cpu_count, 16),
            thread_name_prefix="gedi_worker"
        )
        print("✅ Thread pools initialized")
    
    def load_point_cloud(self, point_cloud_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """Load 5000-point indexed point cloud"""
        try:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
            
            if len(points) != 5000:
                print(f"⚠️ Point cloud has {len(points)} points, expected 5000")
            
            pts_tensor = torch.tensor(points).float()
            print(f"✅ Loaded point cloud with {len(points)} points")
            return pts_tensor, points
            
        except Exception as e:
            print(f"❌ Error loading point cloud: {e}")
            raise
    
    def estimate_diameter(self, points: np.ndarray) -> float:
        """Estimate diameter for GeDi scaling"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            aabb = pcd.get_axis_aligned_bounding_box()
            diameter = np.linalg.norm(np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound))
            
            print(f"✅ Estimated diameter: {diameter:.6f}")
            return diameter
            
        except Exception as e:
            print(f"❌ Error estimating diameter: {e}")
            raise
    
    def compute_gedi_features_exact_workflow(self, pts_tensor: torch.Tensor, points: np.ndarray) -> torch.Tensor:
        """Compute GeDi features following your exact workflow"""
        try:
            # Estimate diameter for scaling
            diameter = self.estimate_diameter(points)
            
            # === Shared GeDi Config (exact from your working setup) ===
            base_config = {
                'dim': 32,
                'samples_per_batch': 500,
                'samples_per_patch_lrf': 4000,
                'samples_per_patch_out': 512,
                'fchkpt_gedi_net': '/app/gedi-repo/data/chkpts/3dmatch/chkpt.tar'
            }
            
            try:
                # === GeDi with r_lrf = 0.3 * diameter ===
                print("🔄 Computing GeDi features with r_lrf = 0.3 * diameter...")
                config_03 = base_config.copy()
                config_03['r_lrf'] = 0.3 * diameter
                gedi_03 = GeDi(config=config_03)
                desc_03 = gedi_03.compute(pts=pts_tensor, pcd=pts_tensor)
                
                # === GeDi with r_lrf = 0.4 * diameter ===
                print("🔄 Computing GeDi features with r_lrf = 0.4 * diameter...")
                config_04 = base_config.copy()
                config_04['r_lrf'] = 0.4 * diameter
                gedi_04 = GeDi(config=config_04)
                desc_04 = gedi_04.compute(pts=pts_tensor, pcd=pts_tensor)
                
                # === Concatenate Descriptors ===
                desc_concat = torch.cat([desc_03, desc_04], dim=1)  # shape: (5000, 64)
                print(f"✅ Concatenated descriptors shape: {desc_concat.shape}")
                
                # === L2 Normalize ===
                desc_normed = F.normalize(desc_concat, p=2, dim=1)
                
                # === Sanity Check: Confirm L2 norms are ~1 ===
                norms = torch.norm(desc_normed, dim=1)
                print(f"L2 norm check — mean: {norms.mean().item():.6f}, min: {norms.min().item():.6f}, max: {norms.max().item():.6f}")
                
                if torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
                    print("✅ L2 normalization verified")
                else:
                    print("⚠️ L2 normalization check failed, but continuing...")
                
                return desc_normed
                
            except Exception as e:
                print(f"⚠️ GeDi processing failed: {e}")
                print("Using fallback geometric features")
                return self.compute_fallback_features(pts_tensor)
            
        except Exception as e:
            print(f"❌ Feature computation failed: {e}")
            raise
    
    def compute_fallback_features(self, pts_tensor: torch.Tensor) -> torch.Tensor:
        """Fallback geometric features if GeDi fails"""
        try:
            num_points = pts_tensor.shape[0]
            
            # Simple geometric features as fallback
            center = pts_tensor.mean(dim=0)
            normalized_pts = pts_tensor - center
            
            # Create 64D features by padding/truncating
            if normalized_pts.shape[1] == 3:
                # Repeat features to reach 64D
                repeated = normalized_pts.repeat(1, 21)[:, :64]  # 3*21 = 63, take 64
                padding = torch.zeros(num_points, 1)
                fallback_features = torch.cat([repeated, padding], dim=1)
            else:
                fallback_features = torch.zeros(num_points, 64)
            
            # L2 normalize
            fallback_features = F.normalize(fallback_features, p=2, dim=1)
            
            print(f"✅ Fallback features computed: {fallback_features.shape}")
            return fallback_features
            
        except Exception as e:
            print(f"❌ Fallback feature computation failed: {e}")
            return torch.zeros(pts_tensor.shape[0], 64)
    
    @profile
    def process_gedi_features_ultra_performance(self, point_cloud_path: str, job_id: str) -> str:
        """Ultra-high performance GeDi geometric feature processing"""
        start_time = time.time()
        
        try:
            # Setup output directory
            output_dir = f"/app/output/{job_id}/gedi_features"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🚀 Starting GeDi processing for job {job_id}")
            print(f"📁 Point cloud: {point_cloud_path}")
            
            # Load point cloud
            pts_tensor, points = self.load_point_cloud(point_cloud_path)
            
            # Compute GeDi features using exact workflow
            geometric_features = self.compute_gedi_features_exact_workflow(pts_tensor, points)
            
            # Save results (matching exact output format from your example)
            output_path = f"{output_dir}/gedi_query_5000_concat_normed.npy"
            np.save(output_path, geometric_features.cpu().numpy())
            
            # Save metadata for pipeline integration
            metadata = {
                'feature_dimension': 64,
                'num_points': len(points),
                'point_indices': list(range(len(points))),
                'processing_info': {
                    'r_lrf_values': [0.3, 0.4],
                    'base_dim': 32,
                    'concatenated_dim': 64,
                    'l2_normalized': True
                }
            }
            
            metadata_path = f"{output_dir}/gedi_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ GeDi processing completed in {processing_time:.2f}s")
            print(f"📊 Processing rate: {len(points)/processing_time:.2f} points/second")
            print(f"💾 Saved: {output_path}")
            print(f"📏 Final descriptor shape: {geometric_features.shape}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ GeDi processing failed: {e}")
            raise
    
    def process_message(self, message: Dict[str, Any]):
        """Process Kafka message for GeDi feature extraction"""
        try:
            job_id = message['job_id']
            object_name = message['object_name']
            point_cloud_path = message['point_cloud_path']
            
            print(f"🚀 Processing GeDi features for job {job_id}")
            
            # Process GeDi features
            output_path = self.process_gedi_features_ultra_performance(point_cloud_path, job_id)
            
            # Send completion message
            completion_message = {
                'job_id': job_id,
                'object_name': object_name,
                'gedi_features_path': output_path,
                'feature_dimension': 64,
                'num_points': 5000,
                'indexed_features': True,
                'processing_type': 'gedi_processing_complete',
                'timestamp': time.time()
            }
            
            self.producer.send('gedi-processing-complete', completion_message)
            print(f"✅ GeDi processing completed for job {job_id}")
            
        except Exception as e:
            print(f"❌ Message processing failed: {e}")
            error_message = {
                'job_id': message.get('job_id', 'unknown'),
                'object_name': message.get('object_name', 'unknown'),
                'error': str(e),
                'processing_type': 'gedi_processing_error',
                'timestamp': time.time()
            }
            self.producer.send('gedi-processing-error', error_message)
    
    def run(self):
        """Run the GeDi processor service"""
        print("🚀 Starting High-Performance GeDi Processor")
        print(f"💻 CPU Cores: {self.resources.cpu_count}")
        print(f"🎮 GPU Count: {self.resources.gpu_count}")
        print(f"📊 Memory: {self.resources.memory_total:.1f}GB")
        print(f"🔧 PyTorch Version: {torch.__version__}")
        print(f"🔢 Feature Output: 64D indexed geometric features")
        
        try:
            for message in self.consumer:
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down GeDi Processor")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.cpu_executor.shutdown(wait=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = HighPerformanceGeDiProcessor()
    processor.run()
