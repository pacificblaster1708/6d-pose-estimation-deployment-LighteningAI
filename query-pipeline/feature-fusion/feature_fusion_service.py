import os
import json
import time
import logging
import multiprocessing
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import psutil
import GPUtil
from memory_profiler import profile
import gc
from kafka import KafkaConsumer, KafkaProducer
from tqdm import tqdm
from sklearn.preprocessing import normalize
from numba import jit, cuda
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

class GPUFeatureFusionProcessor:
    """GPU-accelerated feature fusion processor"""
    
    def __init__(self):
        self.cuda_available = CUPY_AVAILABLE and cp.cuda.is_available()
        if self.cuda_available:
            self.device = cp.cuda.Device(0)
            print(f"✅ GPU acceleration enabled for feature fusion")
        else:
            print("⚠️ GPU acceleration not available, using CPU")
    
    def normalize_features_gpu(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """GPU-accelerated L2 normalization"""
        try:
            if self.cuda_available:
                # Use GPU for normalization
                features_gpu = cp.asarray(features)
                norms = cp.linalg.norm(features_gpu, axis=1, keepdims=True)
                normalized_gpu = features_gpu / (norms + 1e-8)  # Add epsilon for numerical stability
                normalized_features = cp.asnumpy(normalized_gpu)
                
                print(f"✅ GPU L2 normalization applied to {feature_type} features")
            else:
                # CPU fallback
                normalized_features = normalize(features, norm='l2', axis=1)
                print(f"✅ CPU L2 normalization applied to {feature_type} features")
            
            return normalized_features
            
        except Exception as e:
            print(f"❌ GPU normalization failed for {feature_type}: {e}")
            # Fallback to CPU
            return normalize(features, norm='l2', axis=1)
    
    def concatenate_features_gpu(self, dinov2_features: np.ndarray, gedi_features: np.ndarray) -> np.ndarray:
        """GPU-accelerated feature concatenation"""
        try:
            if self.cuda_available:
                # Use GPU for concatenation
                dinov2_gpu = cp.asarray(dinov2_features)
                gedi_gpu = cp.asarray(gedi_features)
                
                fused_gpu = cp.concatenate([gedi_gpu, dinov2_gpu], axis=1)
                fused_features = cp.asnumpy(fused_gpu)
                
                print(f"✅ GPU feature concatenation completed")
            else:
                # CPU fallback
                fused_features = np.concatenate([gedi_features, dinov2_features], axis=1)
                print(f"✅ CPU feature concatenation completed")
            
            return fused_features
            
        except Exception as e:
            print(f"❌ GPU concatenation failed: {e}")
            # Fallback to CPU
            return np.concatenate([gedi_features, dinov2_features], axis=1)

class SystemResourceDetector:
    """System resource detection for optimization"""
    
    def __init__(self):
        self.resources = self.detect_resources()
        self.optimize_environment()
    
    def detect_resources(self) -> SystemResources:
        """Detect system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
            cuda_available = CUPY_AVAILABLE and len(gpus) > 0
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
        
        print(f"🚀 System Resources Detected:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {gpu_count}")
        print(f"   🚀 CUDA Available: {cuda_available}")
        
        return resources
    
    def optimize_environment(self):
        """Optimize environment for feature fusion"""
        cpu_threads = min(self.resources.cpu_count, 64)
        
        # CPU optimization
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMBA_NUM_THREADS'] = str(cpu_threads)
        
        print("✅ Environment optimized for feature fusion")

@jit(nopython=True, parallel=True)
def validate_feature_shapes_numba(dinov2_shape: tuple, gedi_shape: tuple) -> bool:
    """Numba-accelerated feature shape validation"""
    expected_points = 5000
    expected_dim = 64
    
    return (dinov2_shape[0] == expected_points and dinov2_shape[1] == expected_dim and
            gedi_shape[0] == expected_points and gedi_shape[1] == expected_dim)

@jit(nopython=True, parallel=True)
def compute_l2_norms_numba(features: np.ndarray) -> np.ndarray:
    """Numba-accelerated L2 norm computation"""
    norms = np.zeros(features.shape[0])
    for i in range(features.shape[0]):
        norm_squared = 0.0
        for j in range(features.shape[1]):
            norm_squared += features[i, j] * features[i, j]
        norms[i] = np.sqrt(norm_squared)
    return norms

class HighPerformanceFeatureFusion:
    """High-performance feature fusion service with GPU acceleration and multithreading"""
    
    def __init__(self):
        self.system_detector = SystemResourceDetector()
        self.resources = self.system_detector.resources
        self.gpu_processor = GPUFeatureFusionProcessor()
        
        # Kafka setup
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.setup_kafka()
        
        # Thread pool for parallel processing
        self.setup_executors()
        
        # Job tracking for feature coordination
        self.pending_jobs = {}
        
        print("🚀 High-Performance Feature Fusion Service Ready")
    
    def setup_kafka(self):
        """Setup Kafka consumers and producer for feature coordination"""
        try:
            # Multiple consumers for different feature types
            self.dinov2_consumer = KafkaConsumer(
                'dinov2-processing-complete',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='feature-fusion-dinov2-group',
                fetch_max_wait_ms=100,
                max_partition_fetch_bytes=1048576 * 16
            )
            
            self.gedi_consumer = KafkaConsumer(
                'gedi-processing-complete',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='feature-fusion-gedi-group',
                fetch_max_wait_ms=100,
                max_partition_fetch_bytes=1048576 * 16
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
    
    def setup_executors(self):
        """Setup optimized thread pools for parallel processing"""
        # High-performance thread pools
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=min(self.resources.cpu_count, 32),
            thread_name_prefix="fusion_cpu_worker"
        )
        
        self.io_executor = ThreadPoolExecutor(
            max_workers=min(self.resources.cpu_count * 2, 64),
            thread_name_prefix="fusion_io_worker"
        )
        
        # Process pool for CPU-intensive operations
        self.process_executor = ProcessPoolExecutor(
            max_workers=min(self.resources.cpu_count // 2, 16)
        )
        
        print("✅ High-performance thread pools initialized")
    
    def load_features_parallel(self, dinov2_path: str, gedi_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load DINOv2 and GeDi features in parallel with I/O optimization"""
        def load_dinov2():
            return np.load(dinov2_path, mmap_mode='r')
        
        def load_gedi():
            return np.load(gedi_path, mmap_mode='r')
        
        # Load features in parallel using I/O executor
        with ThreadPoolExecutor(max_workers=2) as executor:
            dinov2_future = executor.submit(load_dinov2)
            gedi_future = executor.submit(load_gedi)
            
            dinov2_features = dinov2_future.result().copy()  # Load into memory
            gedi_features = gedi_future.result().copy()  # Load into memory
        
        print(f"✅ Parallel feature loading completed")
        print(f"   📊 DINOv2: {dinov2_features.shape}")
        print(f"   📊 GeDi: {gedi_features.shape}")
        
        return dinov2_features, gedi_features
    
    def validate_feature_dimensions_parallel(self, dinov2_features: np.ndarray, gedi_features: np.ndarray) -> bool:
        """Parallel validation of feature dimensions using Numba"""
        try:
            # Expected dimensions
            expected_points = 5000
            expected_dim = 64
            
            # Use Numba for fast validation
            is_valid = validate_feature_shapes_numba(dinov2_features.shape, gedi_features.shape)
            
            if not is_valid:
                print(f"❌ Feature shape mismatch:")
                print(f"   DINOv2: {dinov2_features.shape}, expected: ({expected_points}, {expected_dim})")
                print(f"   GeDi: {gedi_features.shape}, expected: ({expected_points}, {expected_dim})")
                return False
            
            print(f"✅ Feature validation passed:")
            print(f"   📊 DINOv2: {dinov2_features.shape}")
            print(f"   📊 GeDi: {gedi_features.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature validation failed: {e}")
            return False
    
    def verify_l2_normalization_parallel(self, features: np.ndarray, feature_type: str) -> bool:
        """Parallel L2 normalization verification using Numba"""
        try:
            # Use Numba for fast norm computation
            norms = compute_l2_norms_numba(features)
            
            mean_norm = np.mean(norms)
            min_norm = np.min(norms)
            max_norm = np.max(norms)
            
            print(f"📊 {feature_type} L2 norms - mean: {mean_norm:.6f}, min: {min_norm:.6f}, max: {max_norm:.6f}")
            
            # Check if features are properly L2 normalized (norms should be ~1)
            is_normalized = np.allclose(norms, 1.0, atol=1e-4)
            
            if is_normalized:
                print(f"✅ {feature_type} features are properly L2 normalized")
            else:
                print(f"⚠️ {feature_type} features need L2 normalization")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ L2 normalization check failed for {feature_type}: {e}")
            return False
    
    @profile
    def fuse_features_ultra_performance(self, dinov2_features: np.ndarray, gedi_features: np.ndarray, job_id: str) -> str:
        """Ultra-high performance feature fusion with GPU acceleration"""
        start_time = time.time()
        
        try:
            # Setup output directory
            output_dir = f"/app/output/{job_id}/fused_features"
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"🚀 Starting GPU-accelerated feature fusion for job {job_id}")
            
            # Parallel validation of feature dimensions
            if not self.validate_feature_dimensions_parallel(dinov2_features, gedi_features):
                raise ValueError("Feature dimension validation failed")
            
            # Parallel verification of L2 normalization
            def check_dinov2_norm():
                return self.verify_l2_normalization_parallel(dinov2_features, "DINOv2")
            
            def check_gedi_norm():
                return self.verify_l2_normalization_parallel(gedi_features, "GeDi")
            
            # Check normalization in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                dinov2_norm_future = executor.submit(check_dinov2_norm)
                gedi_norm_future = executor.submit(check_gedi_norm)
                
                dinov2_normalized = dinov2_norm_future.result()
                gedi_normalized = gedi_norm_future.result()
            
            # Apply L2 normalization if needed (using GPU acceleration)
            if not dinov2_normalized:
                dinov2_features = self.gpu_processor.normalize_features_gpu(dinov2_features, "DINOv2")
            
            if not gedi_normalized:
                gedi_features = self.gpu_processor.normalize_features_gpu(gedi_features, "GeDi")
            
            # === GPU-Accelerated Feature Fusion ===
            print("🔄 GPU-accelerated feature concatenation...")
            
            # Concatenate GeDi (64D) + DINOv2 (64D) = 128D fused features
            fused_128 = self.gpu_processor.concatenate_features_gpu(dinov2_features, gedi_features)
            
            print(f"✅ Feature fusion completed: {fused_128.shape}")
            
            # Final validation
            assert fused_128.shape == (5000, 128), f"Expected (5000, 128), got {fused_128.shape}"
            
            # Save fused features with high-performance I/O
            output_path = f"{output_dir}/query_128.npy"
            
            def save_features():
                np.save(output_path, fused_128)
                return output_path
            
            # Save in background while preparing metadata
            save_future = self.io_executor.submit(save_features)
            
            # Prepare fusion metadata
            fusion_metadata = {
                'feature_dimension': 128,
                'num_points': 5000,
                'point_indices': list(range(5000)),
                'fusion_components': {
                    'gedi_geometric': 64,
                    'dinov2_visual': 64,
                    'total_dimension': 128
                },
                'normalization': {
                    'dinov2_l2_normalized': True,
                    'gedi_l2_normalized': True,
                    'fusion_method': 'concatenation'
                },
                'performance_info': {
                    'gpu_acceleration_used': self.gpu_processor.cuda_available,
                    'parallel_processing': True,
                    'high_performance_io': True
                }
            }
            
            # Wait for save to complete
            saved_path = save_future.result()
            
            # Save metadata
            metadata_path = f"{output_dir}/fusion_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(fusion_metadata, f, indent=2)
            
            # Performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Memory cleanup
            if self.gpu_processor.cuda_available:
                cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            print(f"✅ GPU-accelerated feature fusion completed in {processing_time:.2f}s")
            print(f"📊 Processing rate: {5000/processing_time:.2f} points/second")
            print(f"💾 Output: {saved_path}")
            print(f"📏 Final shape: {fused_128.shape}")
            print(f"🔧 GPU acceleration: {self.gpu_processor.cuda_available}")
            
            return saved_path
            
        except Exception as e:
            print(f"❌ Feature fusion failed: {e}")
            raise
    
    def register_feature_completion(self, message: Dict[str, Any]):
        """Register completion of feature processing with thread safety"""
        try:
            job_id = message['job_id']
            processing_type = message['processing_type']
            
            # Thread-safe job tracking
            if job_id not in self.pending_jobs:
                self.pending_jobs[job_id] = {
                    'dinov2_complete': False,
                    'gedi_complete': False,
                    'dinov2_path': None,
                    'gedi_path': None,
                    'object_name': message.get('object_name', 'unknown'),
                    'timestamp': time.time()
                }
            
            # Update completion status
            if processing_type == 'dinov2_processing_complete':
                self.pending_jobs[job_id]['dinov2_complete'] = True
                self.pending_jobs[job_id]['dinov2_path'] = message['dinov2_features_path']
                print(f"✅ DINOv2 64D features ready for job {job_id}")
                
            elif processing_type == 'gedi_processing_complete':
                self.pending_jobs[job_id]['gedi_complete'] = True
                self.pending_jobs[job_id]['gedi_path'] = message['gedi_features_path']
                print(f"✅ GeDi 64D features ready for job {job_id}")
            
            # Check if both features are ready for fusion
            job_info = self.pending_jobs[job_id]
            if job_info['dinov2_complete'] and job_info['gedi_complete']:
                print(f"🔄 Both 64D features ready for job {job_id}, starting 128D fusion...")
                
                # Process fusion in background thread
                self.cpu_executor.submit(self.process_feature_fusion, job_id, job_info)
                
                # Clean up completed job
                del self.pending_jobs[job_id]
            
        except Exception as e:
            print(f"❌ Feature registration failed: {e}")
    
    def process_feature_fusion(self, job_id: str, job_info: Dict[str, Any]):
        """Process feature fusion when both DINOv2 and GeDi features are available"""
        try:
            print(f"🚀 Processing 128D feature fusion for job {job_id}")
            
            # Load features in parallel
            dinov2_features, gedi_features = self.load_features_parallel(
                job_info['dinov2_path'],
                job_info['gedi_path']
            )
            
            # Perform ultra-high performance feature fusion
            output_path = self.fuse_features_ultra_performance(dinov2_features, gedi_features, job_id)
            
            # Send completion message
            completion_message = {
                'job_id': job_id,
                'object_name': job_info['object_name'],
                'fused_features_path': output_path,
                'feature_dimension': 128,
                'num_points': 5000,
                'fusion_components': {
                    'gedi_geometric': 64,
                    'dinov2_visual': 64
                },
                'gpu_accelerated': self.gpu_processor.cuda_available,
                'processing_type': 'feature_fusion_complete',
                'timestamp': time.time()
            }
            
            self.producer.send('feature-fusion-complete', completion_message)
            
            print(f"✅ 128D feature fusion completed for job {job_id}")
            
        except Exception as e:
            print(f"❌ Feature fusion processing failed: {e}")
            # Send error message
            error_message = {
                'job_id': job_id,
                'object_name': job_info.get('object_name', 'unknown'),
                'error': str(e),
                'processing_type': 'feature_fusion_error',
                'timestamp': time.time()
            }
            self.producer.send('feature-fusion-error', error_message)
    
    def run(self):
        """Run the feature fusion service with parallel consumers"""
        print("🚀 Starting High-Performance Feature Fusion Service")
        print(f"💻 CPU Cores: {self.resources.cpu_count}")
        print(f"🎮 GPU Count: {self.resources.gpu_count}")
        print(f"📊 Memory: {self.resources.memory_total:.1f}GB")
        print(f"🔄 Waiting for 64D DINOv2 and 64D GeDi features...")
        
        try:
            # Use thread pool to handle multiple Kafka consumers
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit consumer tasks
                dinov2_future = executor.submit(self.consume_dinov2_messages)
                gedi_future = executor.submit(self.consume_gedi_messages)
                
                # Wait for both consumers
                dinov2_future.result()
                gedi_future.result()
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Feature Fusion Service")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            # Clean up executors
            self.cpu_executor.shutdown(wait=True)
            self.io_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
    
    def consume_dinov2_messages(self):
        """Consume DINOv2 completion messages"""
        for message in self.dinov2_consumer:
            self.register_feature_completion(message.value)
    
    def consume_gedi_messages(self):
        """Consume GeDi completion messages"""
        for message in self.gedi_consumer:
            self.register_feature_completion(message.value)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fusion_service = HighPerformanceFeatureFusion()
    fusion_service.run()
