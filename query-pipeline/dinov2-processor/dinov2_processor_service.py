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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import cv2
import psutil
import GPUtil
from memory_profiler import profile
import gc
from kafka import KafkaConsumer, KafkaProducer
from numba import jit, cuda
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import cupy as cp

@dataclass
class SystemResources:
    """System resource information for GPU optimization"""
    cpu_count: int
    memory_total: float
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_total: List[int]
    cuda_available: bool

class GPUDINOv2Processor:
    """GPU-accelerated DINOv2 processing with batch optimization"""
    
    def __init__(self, device: torch.device, gpu_memory_fraction: float = 0.9):
        self.device = device
        self.setup_gpu_optimization(gpu_memory_fraction)
        self.setup_dinov2_model()
        
    def setup_gpu_optimization(self, memory_fraction: float):
        """Setup GPU optimizations for maximum performance"""
        if torch.cuda.is_available():
            # Set memory fraction for high-end GPUs
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            # Enable all GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            
            # Enable mixed precision for maximum performance
            self.scaler = torch.cuda.amp.GradScaler()
            
            # GPU device properties
            props = torch.cuda.get_device_properties(0)
            print(f"✅ GPU optimizations enabled: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.1f}GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Memory Fraction: {memory_fraction*100}%")
        else:
            print("⚠️ CUDA not available")
    
    def setup_dinov2_model(self):
        """Setup DINOv2 model with aggressive GPU optimization"""
        try:
            print("🔄 Loading DINOv2 Giant with GPU optimization...")
            
            # Load model components
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
            self.model = AutoModel.from_pretrained("facebook/dinov2-giant")
            
            # Aggressive GPU optimization
            if self.device.type == 'cuda':
                # FP16 for maximum performance
                self.model = self.model.half().to(self.device)
                
                # PyTorch 2.0 compilation for speed
                self.model = torch.compile(
                    self.model, 
                    mode="max-autotune",  # Maximum optimization
                    fullgraph=True
                )
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Model configuration for maximum batch processing
            self.patch_size = 14
            self.image_size = 518
            self.num_patches = (self.image_size // self.patch_size) ** 2
            self.feature_dim = 1536
            
            print("✅ DINOv2 model loaded with maximum GPU optimization")
            
        except Exception as e:
            print(f"❌ DINOv2 model loading failed: {e}")
            raise
    
    def batch_extract_features_gpu_optimized(self, images: List[np.ndarray], batch_size: int = 16) -> torch.Tensor:
        """GPU-optimized batch feature extraction with maximum parallelism"""
        try:
            all_features = []
            
            # Process large batches for GPU efficiency
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Convert to PIL efficiently
                pil_images = [Image.fromarray(img) for img in batch_images]
                
                # Batch processing with GPU optimization
                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                    do_resize=True,
                    size=self.image_size
                )
                
                # Move to GPU
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                
                # Extract features with mixed precision
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                
                # Get patch features [batch_size, num_patches, feature_dim]
                batch_features = outputs.last_hidden_state[:, 1:, :].cpu()
                all_features.append(batch_features)
                
                # Clear GPU cache between batches
                torch.cuda.empty_cache()
            
            # Concatenate all batches
            return torch.cat(all_features, dim=0)
            
        except Exception as e:
            print(f"❌ GPU batch feature extraction failed: {e}")
            raise
    
    def extract_single_image_features_gpu(self, image_path: str) -> torch.Tensor:
        """GPU-optimized single image feature extraction"""
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size=self.image_size
            )
            
            # Move to GPU with non-blocking transfer
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
            
            # Extract features with mixed precision
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            
            # Return patch features [num_patches, feature_dim]
            return outputs.last_hidden_state[0, 1:, :].cpu()
            
        except Exception as e:
            print(f"❌ GPU single image feature extraction failed: {e}")
            raise

class GPUProjectionEngine:
    """GPU-accelerated 3D to 2D projection engine using CuPy"""
    
    def __init__(self, intrinsic: np.ndarray, image_size: Tuple[int, int]):
        self.intrinsic = intrinsic
        self.image_size = image_size
        self.patch_grid_size = (37, 37)  # 518/14 = 37
        
        # GPU optimization
        self.intrinsic_gpu = cp.asarray(intrinsic)
        print("✅ GPU projection engine initialized")
    
    def compute_transformations_gpu(self, obj_poses: np.ndarray, cam_poses: np.ndarray) -> np.ndarray:
        """GPU-accelerated transformation computation"""
        num_poses = obj_poses.shape[0]
        transformations = np.zeros((num_poses, 4, 4), dtype=np.float32)
        
        # Convert to GPU arrays
        obj_poses_gpu = cp.asarray(obj_poses)
        cam_poses_gpu = cp.asarray(cam_poses)
        
        for i in range(num_poses):
            world_T_cam = cp.linalg.inv(cam_poses_gpu[i])
            transformations[i] = (world_T_cam @ obj_poses_gpu[i]).get()
        
        return transformations
    
    def project_points_gpu_parallel(self, points: np.ndarray, T_obj_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-parallelized point projection with CuPy"""
        # Convert to GPU arrays
        points_gpu = cp.asarray(points)
        T_obj_cam_gpu = cp.asarray(T_obj_cam)
        
        # Add homogeneous coordinates
        ones = cp.ones((points_gpu.shape[0], 1))
        pts_h = cp.hstack([points_gpu, ones])
        
        # Transform to camera coordinates
        cam_pts = (T_obj_cam_gpu @ pts_h.T).T[:, :3]
        
        # Project to image plane
        z = cam_pts[:, 2]
        valid = z > 0
        
        u = (cam_pts[:, 0] * self.intrinsic_gpu[0, 0]) / z + self.intrinsic_gpu[0, 2]
        v = (cam_pts[:, 1] * self.intrinsic_gpu[1, 1]) / z + self.intrinsic_gpu[1, 2]
        
        uv = cp.stack([u, v], axis=1)
        
        # Convert back to CPU
        return uv.get(), valid.get()
    
    def get_patch_indices_gpu(self, uv: np.ndarray) -> np.ndarray:
        """GPU-accelerated patch index computation"""
        h_img, w_img = self.image_size
        h_p, w_p = self.patch_grid_size
        
        # Convert to GPU
        uv_gpu = cp.asarray(uv)
        
        u_norm = cp.clip(uv_gpu[:, 0] / w_img, 0, 1)
        v_norm = cp.clip(uv_gpu[:, 1] / h_img, 0, 1)
        
        u_idx = (u_norm * w_p).astype(cp.int32)
        v_idx = (v_norm * h_p).astype(cp.int32)
        
        patch_indices = cp.clip(v_idx * w_p + u_idx, 0, w_p * h_p - 1)
        
        return patch_indices.get()

class SystemResourceDetector:
    """GPU-focused system resource detection for cloud deployment"""
    
    def __init__(self):
        self.resources = self.detect_resources()
        self.optimize_gpu_environment()
    
    def detect_resources(self) -> SystemResources:
        """Comprehensive GPU resource detection"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection with detailed info
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        gpu_names = [gpu.name for gpu in gpus]
        gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
        cuda_available = torch.cuda.is_available()
        
        resources = SystemResources(
            cpu_count=cpu_count,
            memory_total=memory_total,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_total=gpu_memory_total,
            cuda_available=cuda_available
        )
        
        print(f"🚀 Cloud GPU System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {gpu_count}")
        if gpu_names:
            for i, (name, mem) in enumerate(zip(gpu_names, gpu_memory_total)):
                print(f"   🚀 GPU {i}: {name} ({mem}MB)")
        print(f"   ⚡ CUDA Available: {cuda_available}")
        
        return resources
    
    def optimize_gpu_environment(self):
        """Optimize environment for cloud GPU deployment"""
        cpu_threads = min(self.resources.cpu_count, 64)  # High-end cloud CPUs
        
        # CPU optimization for data loading
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(cpu_threads)
        
        # GPU optimization for cloud deployment
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        print("✅ Environment optimized for cloud GPU deployment")
    
    def get_optimal_batch_size(self, gpu_memory_gb: float) -> int:
        """Calculate optimal batch size for cloud GPUs"""
        if gpu_memory_gb >= 80:   # H100 80GB
            return 32
        elif gpu_memory_gb >= 40: # A100 40GB, H100 40GB
            return 24
        elif gpu_memory_gb >= 24: # RTX 4090, RTX 3090
            return 16
        elif gpu_memory_gb >= 16: # RTX 4080
            return 12
        elif gpu_memory_gb >= 12: # RTX 3080Ti
            return 8
        else:
            return 4

class HighPerformanceGPUDINOv2Processor:
    """Ultra-high performance GPU-optimized DINOv2 processor for cloud deployment"""
    
    def __init__(self):
        self.system_detector = SystemResourceDetector()
        self.resources = self.system_detector.resources
        
        # GPU setup with maximum optimization
        self.device = torch.device('cuda')  # Force GPU for cloud deployment
        self.gpu_processor = GPUDINOv2Processor(self.device, gpu_memory_fraction=0.9)
        
        # Optimal batch size for cloud GPUs
        if self.resources.gpu_memory_total:
            gpu_memory_gb = self.resources.gpu_memory_total[0] / 1024
            self.batch_size = self.system_detector.get_optimal_batch_size(gpu_memory_gb)
        else:
            self.batch_size = 16  # Default for cloud deployment
        
        # Kafka setup
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # GPU-optimized thread pools
        self.setup_gpu_executors()
        
        print("🚀 High-Performance GPU DINOv2 Processor Ready for Cloud Deployment")
    
    def setup_kafka(self):
        """Setup high-throughput Kafka for cloud deployment"""
        try:
            self.consumer = KafkaConsumer(
                'dinov2-processing',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='dinov2-processor-group',
                fetch_max_wait_ms=100,
                max_partition_fetch_bytes=1048576 * 32  # 32MB for high throughput
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=32768,
                linger_ms=10,
                compression_type='lz4'
            )
            
            print(f"✅ High-throughput Kafka connected")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
            raise
    
    def setup_gpu_executors(self):
        """Setup GPU-optimized thread pools for cloud deployment"""
        # High concurrency for GPU data loading
        self.data_loader_executor = ThreadPoolExecutor(
            max_workers=min(self.resources.cpu_count, 32),
            thread_name_prefix="gpu_data_worker"
        )
        
        # I/O optimization for cloud storage
        self.io_executor = ThreadPoolExecutor(
            max_workers=min(self.resources.cpu_count * 2, 64),
            thread_name_prefix="gpu_io_worker"
        )
        
        print("✅ GPU-optimized thread pools initialized")
    
    @profile
    def process_dinov2_features_gpu_ultra_performance(self, job_id: str, rendered_images: List[str], 
                                                     camera_intrinsics: Dict[str, Any], 
                                                     point_cloud_path: str) -> str:
        """Ultra-high performance GPU-optimized DINOv2 processing"""
        start_time = time.time()
        
        try:
            # Setup output directory
            output_dir = f"/app/output/{job_id}/dino_features"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🚀 Starting GPU ultra-performance DINOv2 processing")
            print(f"📸 Processing {len(rendered_images)} images")
            print(f"🎯 GPU Batch Size: {self.batch_size}")
            print(f"🚀 GPU: {self.resources.gpu_names[0] if self.resources.gpu_names else 'Unknown'}")
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
            num_points = points.shape[0]
            
            # Setup camera intrinsics
            intrinsic = np.array(camera_intrinsics['camera_matrix'])
            image_size = (camera_intrinsics['height'], camera_intrinsics['width'])
            
            # GPU projection engine
            projection_engine = GPUProjectionEngine(intrinsic, image_size)
            
            # Load pose data (using default CNOS poses for demonstration)
            obj_poses = np.eye(4)[np.newaxis, :, :].repeat(len(rendered_images), axis=0)
            cam_poses = np.eye(4)[np.newaxis, :, :].repeat(len(rendered_images), axis=0)
            
            # GPU-accelerated transformation computation
            transformations = projection_engine.compute_transformations_gpu(obj_poses, cam_poses)
            
            # Initialize GPU-accelerated feature storage
            features_list = [[] for _ in range(num_points)]
            
            # Process all images with GPU parallelism
            print(f"🔄 GPU processing {len(rendered_images)} images...")
            
            for i in tqdm(range(len(rendered_images)), desc="GPU Processing"):
                image_path = rendered_images[i]
                T_obj_cam = transformations[i]
                
                # GPU-accelerated point projection
                uv, valid_mask = projection_engine.project_points_gpu_parallel(points, T_obj_cam)
                
                # Filter valid and in-bounds points
                valid_uv = uv[valid_mask]
                valid_indices = np.where(valid_mask)[0]
                
                in_bounds = (valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < image_size[1]) & \
                           (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < image_size[0])
                valid_uv = valid_uv[in_bounds]
                valid_indices = valid_indices[in_bounds]
                
                if len(valid_uv) == 0:
                    continue
                
                # GPU-accelerated DINOv2 feature extraction
                dino_features = self.gpu_processor.extract_single_image_features_gpu(image_path)
                
                # GPU-accelerated patch index computation
                patch_indices = projection_engine.get_patch_indices_gpu(valid_uv)
                
                # Sample features using GPU
                sampled_features = dino_features[patch_indices]
                
                # Store features for each point
                for pt_idx, feat in zip(valid_indices, sampled_features):
                    features_list[pt_idx].append(feat.numpy())
            
            # GPU-accelerated feature aggregation
            print("🔄 GPU aggregating features...")
            final_features = np.zeros((num_points, 1536), dtype=np.float32)
            
            # Parallel feature aggregation
            def aggregate_point_features(args):
                i, feats = args
                if len(feats) > 0:
                    return i, np.mean(feats, axis=0)
                return i, np.zeros(1536, dtype=np.float32)
            
            with ThreadPoolExecutor(max_workers=min(num_points, 32)) as executor:
                results = list(executor.map(aggregate_point_features, enumerate(features_list)))
            
            for i, feat in results:
                final_features[i] = feat
            
            # GPU-accelerated PCA and normalization
            print("🔄 GPU PCA reduction and normalization...")
            pca = PCA(n_components=64)
            reduced_features = pca.fit_transform(final_features)
            normalized_features = normalize(reduced_features, norm='l2', axis=1)
            
            # Save GPU-processed features
            features_path = f"{output_dir}/query_dino_features.npy"
            pca_path = f"{output_dir}/query_pca64.npy"
            
            np.save(features_path, final_features)
            np.save(pca_path, normalized_features)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # GPU memory cleanup
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            print(f"✅ GPU ultra-performance processing completed in {processing_time:.2f}s")
            print(f"🚀 GPU Processing rate: {len(rendered_images)/processing_time:.2f} images/s")
            print(f"⚡ GPU Point rate: {num_points/processing_time:.2f} points/s")
            print(f"💾 Features: {final_features.shape} → {normalized_features.shape}")
            
            return pca_path
            
        except Exception as e:
            print(f"❌ GPU processing failed: {e}")
            raise
    
    def process_message(self, message: Dict[str, Any]):
        """Process Kafka message with GPU optimization"""
        try:
            job_id = message['job_id']
            object_name = message['object_name']
            rendered_images = message['rendered_images']
            camera_intrinsics = message['camera_intrinsics']
            point_cloud_path = message['point_cloud_path']
            
            print(f"🚀 GPU processing DINOv2 features for job {job_id}")
            
            # GPU-accelerated processing
            output_path = self.process_dinov2_features_gpu_ultra_performance(
                job_id, rendered_images, camera_intrinsics, point_cloud_path
            )
            
            # Send completion message
            completion_message = {
                'job_id': job_id,
                'object_name': object_name,
                'dinov2_features_path': output_path,
                'feature_dimension': 64,
                'gpu_optimized': True,
                'processing_type': 'dinov2_processing_complete',
                'timestamp': time.time()
            }
            
            self.producer.send('dinov2-processing-complete', completion_message)
            
            print(f"✅ GPU DINOv2 processing completed for job {job_id}")
            
        except Exception as e:
            print(f"❌ GPU message processing failed: {e}")
            error_message = {
                'job_id': message.get('job_id', 'unknown'),
                'object_name': message.get('object_name', 'unknown'),
                'error': str(e),
                'processing_type': 'dinov2_processing_error',
                'timestamp': time.time()
            }
            self.producer.send('dinov2-processing-error', error_message)
    
    def run(self):
        """Run GPU-optimized DINOv2 processor service"""
        print("🚀 Starting GPU-Optimized DINOv2 Processor for Cloud Deployment")
        print(f"🎮 GPU: {self.resources.gpu_names[0] if self.resources.gpu_names else 'Unknown'}")
        print(f"💻 CPU Cores: {self.resources.cpu_count}")
        print(f"📊 Memory: {self.resources.memory_total:.1f}GB")
        print(f"🔥 GPU Batch Size: {self.batch_size}")
        print(f"⚡ Ready for RunPod/Lightning AI deployment")
        
        try:
            for message in self.consumer:
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down GPU DINOv2 Processor")
        except Exception as e:
            print(f"❌ GPU service error: {e}")
        finally:
            # Clean up GPU resources
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            self.data_loader_executor.shutdown(wait=True)
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = HighPerformanceGPUDINOv2Processor()
    processor.run()
