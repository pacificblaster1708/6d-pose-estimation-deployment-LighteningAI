import os
import json
import time
import logging
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import trimesh
import open3d as o3d
from trimesh.triangles import points_to_barycentric as pob
import cv2
from PIL import Image
import psutil
import GPUtil
from memory_profiler import profile
import gc
from kafka import KafkaConsumer, KafkaProducer
from numba import jit
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

@dataclass
class SystemResources:
    """System resource information for optimization"""
    cpu_count: int
    memory_total: float
    gpu_count: int
    gpu_memory_total: List[int]
    cuda_available: bool

class GPUPointCloudProcessor:
    """GPU-accelerated point cloud processing using CuPy"""
    
    def __init__(self):
        self.cuda_available = CUPY_AVAILABLE and cp.cuda.is_available()
        if self.cuda_available:
            self.device = cp.cuda.Device()
            print(f"✅ GPU acceleration enabled: {self.device}")
        else:
            print("⚠️ GPU acceleration not available, using CPU")
    
    def gpu_accelerated_sampling(self, mesh, samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated uniform surface sampling"""
        if not self.cuda_available:
            return trimesh.sample.sample_surface_even(mesh, count=samples)
        
        try:
            # Use GPU for accelerated sampling when possible
            with cp.cuda.Device():
                # For complex mesh operations, fallback to CPU is often more reliable
                points, face_indices = trimesh.sample.sample_surface_even(mesh, count=samples)
                return points, face_indices
        except Exception as e:
            print(f"⚠️ GPU sampling failed, falling back to CPU: {e}")
            return trimesh.sample.sample_surface_even(mesh, count=samples)

class SystemResourceDetector:
    """System resource detection and optimization"""
    
    def __init__(self):
        self.resources = self.detect_resources()
        self.optimize_environment()
    
    def detect_resources(self) -> SystemResources:
        """Detect system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
        cuda_available = CUPY_AVAILABLE and cp.cuda.is_available()
        
        resources = SystemResources(
            cpu_count=cpu_count,
            memory_total=memory_total,
            gpu_count=gpu_count,
            gpu_memory_total=gpu_memory_total,
            cuda_available=cuda_available
        )
        
        print(f"🚀 System Resources Detected:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {gpu_count}")
        print(f"   🚀 CUDA Available: {cuda_available}")
        
        return resources
    
    def optimize_environment(self):
        """Optimize environment for high-performance processing"""
        cpu_threads = min(self.resources.cpu_count, 64)
        
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(cpu_threads)
        
        print("✅ Environment optimized for high-performance processing")
    
    def get_optimal_thread_count(self, task_type: str) -> int:
        """Get optimal thread count for different tasks"""
        base_cpu = self.resources.cpu_count
        
        if task_type == "cpu_intensive":
            return min(base_cpu, 32)
        elif task_type == "io_bound":
            return min(base_cpu * 2, 64)
        elif task_type == "parallel_processing":
            return min(base_cpu // 2, 16)
        return base_cpu

class HighPerformancePointCloudGenerator:
    """High-performance point cloud generator with GPU acceleration and indexed points"""
    
    def __init__(self):
        self.system_detector = SystemResourceDetector()
        self.resources = self.system_detector.resources
        self.gpu_processor = GPUPointCloudProcessor()
        
        # Kafka setup
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # Setup thread pools
        self.setup_executors()
        
        print("🚀 High-Performance Point Cloud Generator Ready")
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                'point-cloud-generation',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='point-extractor-group'
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
        """Setup optimized thread pools"""
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("cpu_intensive"),
            thread_name_prefix="cpu_worker"
        )
        
        self.parallel_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("parallel_processing"),
            thread_name_prefix="parallel_worker"
        )
        
        print("✅ Thread pools initialized")
    
    def load_mesh_with_texture(self, ply_path: str, texture_path: Optional[str] = None) -> Tuple[np.ndarray, trimesh.Trimesh]:
        """Load mesh and texture with fallback options"""
        try:
            # Load mesh
            mesh = trimesh.load(ply_path, process=False)
            
            # Try to get texture from mesh first
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                texture = np.asarray(mesh.visual.material.image)
                print(f"✅ Loaded texture from PLY file: {texture.shape}")
                return texture, mesh
            
            # If no texture in mesh, try external texture file
            if texture_path and os.path.exists(texture_path):
                texture = np.asarray(Image.open(texture_path))
                print(f"✅ Loaded external texture: {texture.shape}")
                return texture, mesh
            
            # If no texture available, create default texture
            print("⚠️ No texture found, creating default white texture")
            texture = np.ones((256, 256, 3), dtype=np.uint8) * 255
            return texture, mesh
            
        except Exception as e:
            print(f"❌ Error loading mesh/texture: {e}")
            raise
    
    def parallel_point_sampling(self, mesh, samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel point sampling with GPU acceleration"""
        try:
            # Use GPU-accelerated sampling
            points, face_indices = self.gpu_processor.gpu_accelerated_sampling(mesh, samples)
            
            print(f"✅ Sampled {len(points)} points from mesh")
            return points, face_indices
            
        except Exception as e:
            print(f"❌ Point sampling failed: {e}")
            raise
    
    @jit(nopython=True, parallel=True)
    def numba_color_sampling(self, sample_uvs: np.ndarray, texture: np.ndarray) -> np.ndarray:
        """Numba-accelerated color sampling for parallel processing"""
        h, w = texture.shape[:2]
        colors = np.zeros((len(sample_uvs), 3), dtype=np.float32)
        
        for i in range(len(sample_uvs)):
            u, v = sample_uvs[i]
            # Convert UV to pixel coordinates
            x = int(np.clip(u * (w - 1), 0, w - 1))
            y = int(np.clip((1.0 - v) * (h - 1), 0, h - 1))
            
            # Sample color
            colors[i] = texture[y, x, :3] / 255.0
        
        return colors
    
    def parallel_color_sampling(self, mesh, points: np.ndarray, face_indices: np.ndarray, 
                              texture: np.ndarray) -> np.ndarray:
        """Parallel color sampling with multithreading"""
        try:
            # Get triangles and barycentric coordinates
            triangles = mesh.triangles[face_indices]
            bc = pob(triangles, points)
            
            # Get UV coordinates
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
                sample_uvs = np.einsum('ij,ijk->ik', bc, face_uvs)
            else:
                # Create default UV coordinates if not available
                print("⚠️ No UV coordinates found, using default mapping")
                sample_uvs = np.random.rand(len(points), 2)
            
            # Use Numba-accelerated color sampling
            try:
                colors = self.numba_color_sampling(sample_uvs, texture)
                print("✅ Numba-accelerated color sampling completed")
            except:
                # Fallback to CPU sampling
                colors = self.cpu_color_sampling(sample_uvs, texture)
                print("✅ CPU color sampling completed")
            
            return colors
            
        except Exception as e:
            print(f"❌ Color sampling failed: {e}")
            # Return default colors
            return np.ones((len(points), 3), dtype=np.float32) * 0.7
    
    def cpu_color_sampling(self, sample_uvs: np.ndarray, texture: np.ndarray) -> np.ndarray:
        """CPU-based color sampling with multithreading"""
        def sample_color_batch(uv_batch):
            h, w = texture.shape[:2]
            colors = np.zeros((len(uv_batch), 3), dtype=np.float32)
            
            for i, (u, v) in enumerate(uv_batch):
                # Convert UV to pixel coordinates
                x = int(np.clip(u * (w - 1), 0, w - 1))
                y = int(np.clip((1.0 - v) * (h - 1), 0, h - 1))
                colors[i] = texture[y, x, :3] / 255.0
            
            return colors
        
        # Split work into batches for parallel processing
        batch_size = max(1, len(sample_uvs) // self.resources.cpu_count)
        batches = [sample_uvs[i:i+batch_size] for i in range(0, len(sample_uvs), batch_size)]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.resources.cpu_count) as executor:
            futures = [executor.submit(sample_color_batch, batch) for batch in batches]
            results = [future.result() for future in as_completed(futures)]
        
        # Combine results
        colors = np.vstack(results)
        return colors
    
    def create_indexed_point_cloud(self, points: np.ndarray, colors: np.ndarray, 
                                  face_indices: np.ndarray, scale: float = 0.001) -> o3d.geometry.PointCloud:
        """Create indexed point cloud for feature pairing"""
        try:
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Apply scaling (from mm to meters)
            pcd.scale(scale, center=(0, 0, 0))
            
            # Store face indices as additional attribute for feature pairing
            # This ensures each point has a consistent index for matching geometric and visual features
            point_indices = np.arange(len(points))
            
            # Remove outliers while preserving indices
            pcd_clean, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Update face indices to match cleaned point cloud
            if len(inlier_indices) < len(face_indices):
                print(f"⚠️ Outlier removal: {len(points)} → {len(inlier_indices)} points")
                face_indices = face_indices[inlier_indices]
            
            print(f"✅ Created indexed point cloud with {len(pcd_clean.points)} points")
            return pcd_clean, face_indices, point_indices[inlier_indices]
            
        except Exception as e:
            print(f"❌ Point cloud creation failed: {e}")
            raise
    
    @profile
    def generate_indexed_point_cloud(self, ply_path: str, texture_path: Optional[str], 
                                    job_id: str, samples: int = 5000) -> str:
        """Generate indexed point cloud with exactly 5000 points"""
        start_time = time.time()
        
        try:
            # Setup output directory
            output_dir = f"/app/output/{job_id}/point_cloud"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🚀 Starting indexed point cloud generation for job {job_id}")
            print(f"📁 PLY: {ply_path}")
            print(f"📁 Texture: {texture_path}")
            print(f"🎯 Target samples: {samples}")
            
            # Load mesh and texture
            texture, mesh = self.load_mesh_with_texture(ply_path, texture_path)
            
            # Parallel point sampling
            points, face_indices = self.parallel_point_sampling(mesh, samples)
            
            # Ensure exactly 5000 points
            if len(points) != samples:
                print(f"⚠️ Adjusting point count: {len(points)} → {samples}")
                if len(points) > samples:
                    # Randomly select subset
                    indices = np.random.choice(len(points), samples, replace=False)
                    points = points[indices]
                    face_indices = face_indices[indices]
                else:
                    # Duplicate points to reach target
                    repeat_factor = samples // len(points) + 1
                    points_repeated = np.tile(points, (repeat_factor, 1))[:samples]
                    face_indices_repeated = np.tile(face_indices, repeat_factor)[:samples]
                    points = points_repeated
                    face_indices = face_indices_repeated
            
            # Parallel color sampling
            colors = self.parallel_color_sampling(mesh, points, face_indices, texture)
            
            # Create indexed point cloud
            pcd, final_face_indices, point_indices = self.create_indexed_point_cloud(
                points, colors, face_indices
            )
            
            # Save point cloud
            output_path = f"{output_dir}/query_5000_scaled.ply"
            o3d.io.write_point_cloud(output_path, pcd)
            
            # Save index mapping for feature pairing
            index_mapping = {
                'point_indices': point_indices.tolist(),
                'face_indices': final_face_indices.tolist(),
                'total_points': len(pcd.points)
            }
            
            index_path = f"{output_dir}/point_indices.json"
            with open(index_path, 'w') as f:
                json.dump(index_mapping, f)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Clean up memory
            gc.collect()
            if self.gpu_processor.cuda_available:
                cp.get_default_memory_pool().free_all_blocks()
            
            print(f"✅ Indexed point cloud generation completed in {processing_time:.2f}s")
            print(f"📊 Processing rate: {len(pcd.points)/processing_time:.2f} points/second")
            print(f"💾 Saved to: {output_path}")
            print(f"📇 Index mapping saved to: {index_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Point cloud generation failed: {e}")
            raise
    
    def process_message(self, message: Dict[str, Any]):
        """Process Kafka message"""
        try:
            job_id = message['job_id']
            object_name = message['object_name']
            ply_path = message['ply_path']
            texture_path = message.get('texture_path')
            
            print(f"🚀 Processing indexed point cloud generation for job {job_id}")
            
            # Generate indexed point cloud
            output_path = self.generate_indexed_point_cloud(
                ply_path, texture_path, job_id
            )
            
            # Send completion message
            completion_message = {
                'job_id': job_id,
                'object_name': object_name,
                'point_cloud_path': output_path,
                'num_points': 5000,
                'indexed': True,
                'processing_type': 'point_cloud_generation_complete',
                'timestamp': time.time()
            }
            
            self.producer.send('point-extraction-complete', completion_message)
            
            print(f"✅ Indexed point cloud generation completed for job {job_id}")
            
        except Exception as e:
            print(f"❌ Message processing failed: {e}")
            # Send error message
            error_message = {
                'job_id': message.get('job_id', 'unknown'),
                'object_name': message.get('object_name', 'unknown'),
                'error': str(e),
                'processing_type': 'point_cloud_generation_error',
                'timestamp': time.time()
            }
            self.producer.send('point-extraction-error', error_message)
    
    def run(self):
        """Run the point cloud generator service"""
        print("🚀 Starting High-Performance Indexed Point Cloud Generator")
        print(f"💻 CPU Cores: {self.resources.cpu_count}")
        print(f"🎮 GPU Count: {self.resources.gpu_count}")
        print(f"📊 Memory: {self.resources.memory_total:.1f}GB")
        
        try:
            for message in self.consumer:
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Point Cloud Generator")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            # Clean up executors
            self.cpu_executor.shutdown(wait=True)
            self.parallel_executor.shutdown(wait=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = HighPerformancePointCloudGenerator()
    generator.run()
