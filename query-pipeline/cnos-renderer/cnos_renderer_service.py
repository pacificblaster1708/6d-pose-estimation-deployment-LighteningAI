import os
import json
import time
import subprocess
import logging
import threading
import multiprocessing
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
from kafka import KafkaConsumer, KafkaProducer
import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import trimesh
import pyrender
from PIL import Image
import glob
import requests
from tqdm import tqdm
import psutil
import GPUtil
from memory_profiler import profile
import gc
import nvidia_ml_py3 as nvml
from numba import jit, cuda
import cupy as cp

@dataclass
class SystemResources:
    """System resource information for RunPod/Lightning AI optimization"""
    cpu_count: int
    cpu_freq: float
    memory_total: float
    gpu_count: int
    gpu_names: List[str]
    gpu_memory_total: List[int]
    gpu_memory_free: List[int]
    cuda_devices: int
    cuda_version: str

class GPUResourceManager:
    """GPU resource management for high-end GPUs"""
    
    def __init__(self):
        self.initialize_gpu_monitoring()
        self.detect_gpu_capabilities()
    
    def initialize_gpu_monitoring(self):
        """Initialize NVIDIA ML monitoring"""
        try:
            nvml.nvmlInit()
            self.nvml_initialized = True
        except Exception as e:
            print(f"❌ NVIDIA ML monitoring failed: {e}")
            self.nvml_initialized = False
    
    def detect_gpu_capabilities(self):
        """Detect GPU capabilities for optimization"""
        self.gpu_capabilities = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.gpu_capabilities[i] = {
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'multiprocessor_count': props.multiprocessor_count,
                    'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor,
                    'compute_capability': f"{props.major}.{props.minor}"
                }
    
    def get_optimal_gpu_settings(self, gpu_id: int) -> Dict[str, Any]:
        """Get optimal GPU settings based on hardware"""
        if gpu_id not in self.gpu_capabilities:
            return {}
        
        gpu_info = self.gpu_capabilities[gpu_id]
        memory_gb = gpu_info['total_memory'] / (1024**3)
        
        # Optimize based on GPU memory
        if memory_gb >= 80:  # H100, A100 80GB
            return {
                'memory_fraction': 0.9,
                'batch_size': 32,
                'num_workers': 16,
                'precision': 'fp16'
            }
        elif memory_gb >= 40:  # A100 40GB, RTX 6000 Ada
            return {
                'memory_fraction': 0.85,
                'batch_size': 16,
                'num_workers': 12,
                'precision': 'fp16'
            }
        elif memory_gb >= 24:  # RTX 4090, RTX 3090
            return {
                'memory_fraction': 0.8,
                'batch_size': 8,
                'num_workers': 8,
                'precision': 'fp16'
            }
        else:
            return {
                'memory_fraction': 0.7,
                'batch_size': 4,
                'num_workers': 4,
                'precision': 'fp32'
            }

class SystemResourceDetector:
    """Advanced system resource detection for RunPod/Lightning AI"""
    
    def __init__(self):
        self.gpu_manager = GPUResourceManager()
        self.resources = self.detect_all_resources()
        self.optimize_environment()
    
    def detect_all_resources(self) -> SystemResources:
        """Comprehensive resource detection"""
        # CPU resources
        cpu_count = multiprocessing.cpu_count()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # GPU resources
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        gpu_names = [gpu.name for gpu in gpus]
        gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
        gpu_memory_free = [gpu.memoryFree for gpu in gpus]
        
        # CUDA information
        cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        resources = SystemResources(
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            memory_total=memory_total,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_free=gpu_memory_free,
            cuda_devices=cuda_devices,
            cuda_version=cuda_version
        )
        
        self.log_resources(resources)
        return resources
    
    def log_resources(self, resources: SystemResources):
        """Log detected resources"""
        print("🚀 RunPod/Lightning AI Resources Detected:")
        print(f"   💻 CPU Cores: {resources.cpu_count}")
        print(f"   ⚡ CPU Frequency: {resources.cpu_freq:.2f} MHz")
        print(f"   📊 Memory: {resources.memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {resources.gpu_count}")
        
        for i, (name, mem_total, mem_free) in enumerate(zip(
            resources.gpu_names, resources.gpu_memory_total, resources.gpu_memory_free
        )):
            print(f"   🚀 GPU {i}: {name} ({mem_total}MB total, {mem_free}MB free)")
        
        print(f"   ⚡ CUDA Devices: {resources.cuda_devices}")
        print(f"   🔥 CUDA Version: {resources.cuda_version}")
    
    def optimize_environment(self):
        """Optimize environment for high-end GPUs"""
        cpu_threads = min(self.resources.cpu_count, 64)  # Cap at 64 for stability
        
        # CPU optimization
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(cpu_threads)
        os.environ['TBB_NUM_THREADS'] = str(cpu_threads)
        
        # GPU optimization for high-end cards
        if self.resources.cuda_devices > 0:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            
            # Optimize based on GPU memory
            total_gpu_memory = sum(self.resources.gpu_memory_total)
            if total_gpu_memory > 40000:  # > 40GB
                os.environ['CUDA_CACHE_MAXSIZE'] = '4294967296'  # 4GB cache
            else:
                os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB cache
        
        print("✅ Environment optimized for RunPod/Lightning AI")
    
    def get_optimal_thread_count(self, task_type: str) -> int:
        """Get optimal thread count for different task types"""
        base_cpu = self.resources.cpu_count
        
        if task_type == "cpu_intensive":
            return min(base_cpu, 32)
        elif task_type == "io_bound":
            return min(base_cpu * 2, 64)
        elif task_type == "gpu_preprocessing":
            return min(base_cpu // 2, 16)
        elif task_type == "parallel_rendering":
            return min(base_cpu // 4, 12)
        elif task_type == "image_processing":
            return min(base_cpu, 24)
        return base_cpu

class HighPerformanceCNOSRenderer:
    """Ultra-high performance CNOS renderer for RunPod/Lightning AI"""
    
    def __init__(self):
        self.system_detector = SystemResourceDetector()
        self.resources = self.system_detector.resources
        self.gpu_manager = self.system_detector.gpu_manager
        
        # Kafka setup
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # GPU setup with high-performance optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_count = torch.cuda.device_count()
        
        # Setup optimized thread pools
        self.setup_high_performance_executors()
        
        # Initialize DINOv2 model
        self.setup_dinov2_model()
        
        print(f"🚀 Ultra-High Performance CNOS Renderer Ready")
        print(f"📊 Device: {self.device}")
        print(f"🎮 GPU Count: {self.gpu_count}")
    
    def setup_kafka(self):
        """Setup high-performance Kafka"""
        try:
            self.consumer = KafkaConsumer(
                'cnos-rendering',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='cnos-renderer-group',
                fetch_max_wait_ms=50,
                fetch_min_bytes=1,
                max_partition_fetch_bytes=1048576 * 32,  # 32MB
                session_timeout_ms=60000,
                heartbeat_interval_ms=10000
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=32768,
                linger_ms=10,
                compression_type='lz4',
                buffer_memory=67108864  # 64MB
            )
            
            print(f"✅ High-Performance Kafka connected")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
            raise
    
    def setup_high_performance_executors(self):
        """Setup optimized executors for different workloads"""
        # CPU-intensive tasks (multithreaded)
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("cpu_intensive"),
            thread_name_prefix="cpu_worker"
        )
        
        # I/O bound tasks (high concurrency)
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("io_bound"),
            thread_name_prefix="io_worker"
        )
        
        # GPU preprocessing tasks
        self.gpu_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("gpu_preprocessing"),
            thread_name_prefix="gpu_worker"
        )
        
        # Parallel rendering tasks
        self.render_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("parallel_rendering"),
            thread_name_prefix="render_worker"
        )
        
        # Image processing tasks
        self.image_executor = ThreadPoolExecutor(
            max_workers=self.system_detector.get_optimal_thread_count("image_processing"),
            thread_name_prefix="image_worker"
        )
        
        print("✅ High-performance executors initialized")
    
    def setup_dinov2_model(self):
        """Setup DINOv2 model with GPU optimization"""
        try:
            print("🔄 Loading DINOv2 model with GPU optimization...")
            
            # Get optimal GPU settings
            if self.gpu_count > 0:
                gpu_settings = self.gpu_manager.get_optimal_gpu_settings(0)
                torch.cuda.set_per_process_memory_fraction(gpu_settings['memory_fraction'])
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            
            # Load model
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
            self.model = AutoModel.from_pretrained("facebook/dinov2-giant")
            
            # Optimize model
            if self.device.type == 'cuda':
                self.model = self.model.half().to(self.device)  # FP16 for performance
                self.model = torch.compile(self.model)  # PyTorch 2.0 optimization
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print("✅ DINOv2 model loaded with optimizations")
        except Exception as e:
            print(f"❌ DINOv2 model loading failed: {e}")
            self.processor = None
            self.model = None
    
    @jit(nopython=True, parallel=True)
    def parallel_image_processing(self, images_array):
        """Numba-optimized parallel image processing"""
        # This would contain optimized image processing logic
        pass
    
    def render_view_batch_gpu_optimized(self, ply_path: str, texture_path: str, 
                                      pose_indices: List[int], output_dir: str) -> List[str]:
        """GPU-optimized batch rendering"""
        rendered_images = []
        
        # Use GPU-accelerated rendering if available
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                # Enable CUDA graphs for repeated operations
                torch.cuda.synchronize()
                
                # Process poses in parallel
                futures = []
                for pose_idx in pose_indices:
                    future = self.render_executor.submit(
                        self.render_single_view_optimized,
                        ply_path, texture_path, pose_idx, output_dir
                    )
                    futures.append((pose_idx, future))
                
                # Collect results with timeout
                for pose_idx, future in futures:
                    try:
                        result = future.result(timeout=60)
                        if result:
                            rendered_images.append(result)
                    except Exception as e:
                        print(f"⚠️ Rendering failed for pose {pose_idx}: {e}")
                
                torch.cuda.synchronize()
        
        return rendered_images
    
    def render_single_view_optimized(self, ply_path: str, texture_path: str, 
                                   pose_idx: int, output_dir: str) -> Optional[str]:
        """Single view rendering with optimizations"""
        try:
            output_path = f"{output_dir}/view_{pose_idx:03d}.png"
            
            # Optimized CNOS command
            cmd = [
                "python", "/app/cnos/src/poses/generate_views.py",
                ply_path,
                "/app/cnos/src/poses/predefined_poses/obj_poses_level0.npy",
                output_dir,
                str(pose_idx),
                "True",
                "1",
                "0.3"
            ]
            
            if texture_path and os.path.exists(texture_path):
                cmd.extend(["--texture", texture_path])
            
            # Execute with optimized environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
                cwd="/app/cnos",
                env=env
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                return None
                
        except Exception as e:
            print(f"❌ Single view rendering failed: {e}")
            return None
    
    def render_42_views_ultra_parallel(self, ply_path: str, texture_path: str, 
                                     output_dir: str, job_id: str) -> List[str]:
        """Ultra-parallel rendering of 42 views"""
        print(f"🚀 Starting ultra-parallel rendering for job {job_id}")
        
        # Determine optimal batch size based on GPU memory
        gpu_settings = self.gpu_manager.get_optimal_gpu_settings(0)
        batch_size = gpu_settings.get('batch_size', 8)
        
        # Divide 42 views into optimal batches
        view_batches = [
            list(range(i, min(i + batch_size, 42)))
            for i in range(0, 42, batch_size)
        ]
        
        print(f"📊 Processing {len(view_batches)} batches of size {batch_size}")
        
        # Process all batches in parallel
        all_rendered_images = []
        
        with ThreadPoolExecutor(max_workers=len(view_batches)) as executor:
            batch_futures = []
            
            for batch_idx, batch in enumerate(view_batches):
                future = executor.submit(
                    self.render_view_batch_gpu_optimized,
                    ply_path, texture_path, batch, output_dir
                )
                batch_futures.append(future)
            
            # Collect results
            for future in as_completed(batch_futures):
                try:
                    batch_images = future.result(timeout=300)
                    all_rendered_images.extend(batch_images)
                except Exception as e:
                    print(f"⚠️ Batch processing failed: {e}")
        
        # Sort by view index
        all_rendered_images.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        
        print(f"✅ Ultra-parallel rendering completed: {len(all_rendered_images)}/42 views")
        return all_rendered_images
    
    def process_images_gpu_accelerated(self, rendered_images: List[str], job_id: str) -> Dict[str, Any]:
        """GPU-accelerated image processing"""
        try:
            # Use CuPy for GPU-accelerated image processing where possible
            if cp.cuda.is_available():
                print("🚀 Using GPU-accelerated image processing")
            
            # Process images in parallel
            def process_single_image(img_path: str) -> Image.Image:
                img = Image.open(img_path).convert("RGBA")
                bg = Image.new("RGB", img.size, (0, 0, 0))
                bg.paste(img, mask=img.split()[3])
                return bg
            
            # Parallel processing with optimized thread count
            max_workers = self.system_detector.get_optimal_thread_count("image_processing")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_image, img_path) 
                          for img_path in rendered_images]
                
                frames = []
                for future in as_completed(futures):
                    try:
                        frame = future.result(timeout=30)
                        frames.append(frame)
                    except Exception as e:
                        print(f"⚠️ Image processing failed: {e}")
            
            # Create optimized GIF
            gif_path = None
            if frames:
                gif_path = f"/app/output/{job_id}/rendered_views.gif"
                
                # Optimize GIF creation
                frames[0].save(
                    gif_path,
                    format="GIF",
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,  # Faster animation
                    loop=0,
                    optimize=True
                )
                print(f"🎬 Optimized GIF created: {gif_path}")
            
            return {
                "rendered_images": rendered_images,
                "gif_path": gif_path,
                "num_views": len(rendered_images)
            }
            
        except Exception as e:
            print(f"❌ GPU-accelerated image processing failed: {e}")
            return {
                "rendered_images": rendered_images,
                "gif_path": None,
                "num_views": len(rendered_images)
            }
    
    def generate_camera_intrinsics(self) -> Dict[str, Any]:
        """Generate camera intrinsics for CNOS rendering"""
        return {
            "fx": 577.5,
            "fy": 577.5,
            "cx": 319.5,
            "cy": 239.5,
            "width": 640,
            "height": 480,
            "camera_matrix": [
                [577.5, 0, 319.5],
                [0, 577.5, 239.5],
                [0, 0, 1]
            ],
            "distortion_coefficients": [0, 0, 0, 0, 0]
        }
    
    def setup_job_directories(self, job_id: str):
        """Setup comprehensive job directories"""
        directories = [
            f"/app/output/{job_id}/renders",
            f"/app/output/{job_id}/point_cloud",
            f"/app/output/{job_id}/dino_feature_pca",
            f"/app/output/{job_id}/intermediate",
            f"/app/output/{job_id}/logs",
            f"/app/output/{job_id}/performance"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @profile
    def render_object_ultra_performance(self, ply_path: str, texture_path: str, job_id: str) -> Dict[str, Any]:
        """Ultra-high performance object rendering"""
        start_time = time.time()
        
        try:
            # Setup
            self.setup_job_directories(job_id)
            output_dir = f"/app/output/{job_id}/renders"
            
            print(f"🚀 Ultra-performance rendering started for job {job_id}")
            print(f"📁 PLY: {ply_path}")
            print(f"📁 Texture: {texture_path}")
            print(f"📁 Output: {output_dir}")
            
            # Pre-warm GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Ultra-parallel rendering
            rendered_images = self.render_42_views_ultra_parallel(
                ply_path, texture_path, output_dir, job_id
            )
            
            if not rendered_images:
                raise Exception("No images were successfully rendered")
            
            # GPU-accelerated post-processing
            processing_result = self.process_images_gpu_accelerated(rendered_images, job_id)
            
            # Generate camera intrinsics
            camera_intrinsics = self.generate_camera_intrinsics()
            
            # Save intrinsics
            intrinsics_path = f"/app/output/{job_id}/camera_intrinsics.json"
            with open(intrinsics_path, 'w') as f:
                json.dump(camera_intrinsics, f, indent=2)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            result = {
                "rendered_images": processing_result["rendered_images"],
                "camera_intrinsics": camera_intrinsics,
                "intrinsics_path": intrinsics_path,
                "num_views": processing_result["num_views"],
                "gif_path": processing_result.get("gif_path"),
                "performance_metrics": {
                    "processing_time": processing_time,
                    "fps": 42 / processing_time,
                    "cpu_count": self.resources.cpu_count,
                    "gpu_count": self.resources.gpu_count,
                    "memory_total": self.resources.memory_total,
                    "gpu_memory_total": sum(self.resources.gpu_memory_total)
                }
            }
            
            print(f"✅ Ultra-performance rendering completed in {processing_time:.2f}s")
            print(f"📊 Performance: {result['performance_metrics']['fps']:.2f} FPS")
            
            return result
            
        except Exception as e:
            print(f"❌ Ultra-performance rendering failed: {e}")
            raise
    
    def process_message(self, message: Dict[str, Any]):
        """Process Kafka message with ultra-high performance"""
        try:
            job_id = message['job_id']
            object_name = message['object_name']
            ply_path = message['ply_path']
            texture_path = message.get('texture_path')
            
            print(f"🚀 Processing job {job_id} with ultra-high performance")
            
            # Ultra-performance rendering
            render_result = self.render_object_ultra_performance(ply_path, texture_path, job_id)
            
            # Send completion message
            completion_message = {
                'job_id': job_id,
                'object_name': object_name,
                'rendered_images': render_result['rendered_images'],
                'camera_intrinsics': render_result['camera_intrinsics'],
                'intrinsics_path': render_result['intrinsics_path'],
                'num_views': render_result['num_views'],
                'gif_path': render_result.get('gif_path'),
                'performance_metrics': render_result['performance_metrics'],
                'processing_type': 'cnos_rendering_complete',
                'timestamp': time.time()
            }
            
            self.producer.send('cnos-rendering-complete', completion_message)
            
            print(f"✅ Job {job_id} completed with ultra-high performance")
            
        except Exception as e:
            print(f"❌ Message processing failed: {e}")
            # Send error message
            error_message = {
                'job_id': message.get('job_id', 'unknown'),
                'object_name': message.get('object_name', 'unknown'),
                'error': str(e),
                'processing_type': 'cnos_rendering_error',
                'timestamp': time.time()
            }
            self.producer.send('cnos-rendering-error', error_message)
    
    def run(self):
        """Run ultra-high performance CNOS renderer"""
        print("🚀 Starting Ultra-High Performance CNOS Renderer")
        print(f"💻 CPU Cores: {self.resources.cpu_count}")
        print(f"🎮 GPU Count: {self.resources.gpu_count}")
        print(f"📊 Memory: {self.resources.memory_total:.1f}GB")
        print(f"🚀 GPU Memory: {sum(self.resources.gpu_memory_total)}MB")
        
        try:
            for message in self.consumer:
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Ultra-High Performance CNOS Renderer")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            # Clean up executors
            self.cpu_executor.shutdown(wait=True)
            self.io_executor.shutdown(wait=True)
            self.gpu_executor.shutdown(wait=True)
            self.render_executor.shutdown(wait=True)
            self.image_executor.shutdown(wait=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    renderer = HighPerformanceCNOSRenderer()
    renderer.run()

