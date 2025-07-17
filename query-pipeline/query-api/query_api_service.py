import os
import json
import time
import logging
import multiprocessing
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import asyncio
import psutil
import GPUtil
from memory_profiler import profile
import gc

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import uvicorn
import numpy as np
from PIL import Image
import cv2

@dataclass
class SystemResources:
    """System resource information for optimization"""
    cpu_count: int
    memory_total: float
    gpu_count: int
    gpu_memory_total: List[int]

class QueryProcessingJob(BaseModel):
    job_id: str
    object_name: str
    has_texture: bool
    status: str
    created_at: float
    processing_time: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None

class SystemResourceManager:
    """Manages system resources and optimizes performance"""
    
    def __init__(self):
        self.resources = self.detect_resources()
        self.optimize_environment()
        
    def detect_resources(self) -> SystemResources:
        """Detect available system resources"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        try:
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
        except:
            gpu_count = 0
            gpu_memory_total = []
        
        resources = SystemResources(
            cpu_count=cpu_count,
            memory_total=memory_total,
            gpu_count=gpu_count,
            gpu_memory_total=gpu_memory_total
        )
        
        print(f"🚀 System Resources Detected:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {gpu_count}")
        
        return resources
    
    def optimize_environment(self):
        """Optimize environment for high-performance processing"""
        cpu_threads = min(self.resources.cpu_count, 32)
        
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        
        print(f"✅ Environment optimized for {cpu_threads} threads")
    
    def get_optimal_thread_count(self, task_type: str) -> int:
        """Get optimal thread count for different task types"""
        base_cpu = self.resources.cpu_count
        
        if task_type == "file_processing":
            return min(base_cpu, 8)
        elif task_type == "image_processing":
            return min(base_cpu, 16)
        elif task_type == "io_operations":
            return min(base_cpu * 2, 32)
        return base_cpu

class HighPerformanceQueryAPI:
    """High-performance Query API with multithreading and GPU optimization"""
    
    def __init__(self):
        self.app = FastAPI(
            title="High-Performance Query Pipeline API",
            version="2.0.0",
            description="GPU-accelerated API for 6D pose estimation pipeline"
        )
        
        # Initialize system resources
        self.resource_manager = SystemResourceManager()
        
        # Setup CORS and middleware
        self.setup_cors()
        self.setup_endpoints()
        
        # Kafka setup
        self.producer = None
        self.kafka_ready = False
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        
        # Job tracking with thread-safe operations
        self.active_jobs = {}
        self.job_lock = asyncio.Lock()
        
        # Setup thread pools for different operations
        self.setup_thread_pools()
        
        # Setup directories
        self.setup_directories()
        
        # Initialize Kafka
        self.wait_for_kafka()
        
    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_thread_pools(self):
        """Setup optimized thread pools for different operations"""
        # File processing thread pool
        self.file_executor = ThreadPoolExecutor(
            max_workers=self.resource_manager.get_optimal_thread_count("file_processing"),
            thread_name_prefix="file_worker"
        )
        
        # Image processing thread pool
        self.image_executor = ThreadPoolExecutor(
            max_workers=self.resource_manager.get_optimal_thread_count("image_processing"),
            thread_name_prefix="image_worker"
        )
        
        # I/O operations thread pool
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.resource_manager.get_optimal_thread_count("io_operations"),
            thread_name_prefix="io_worker"
        )
        
        print("✅ Thread pools initialized for optimal performance")
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ["/app/uploads", "/app/data", "/app/output", "/app/logs"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def wait_for_kafka(self, max_retries=30, retry_delay=5):
        """Wait for Kafka to be available with optimized connection"""
        for attempt in range(max_retries):
            try:
                print(f"🔄 Connecting to Kafka at {self.kafka_servers} (attempt {attempt + 1}/{max_retries})")
                
                self.producer = KafkaProducer(
                    bootstrap_servers=[self.kafka_servers],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    max_block_ms=10000,
                    retries=3,
                    retry_backoff_ms=1000,
                    batch_size=16384,
                    linger_ms=10,
                    compression_type='snappy'
                )
                
                self.kafka_ready = True
                print(f"✅ Successfully connected to Kafka at {self.kafka_servers}!")
                return
                
            except NoBrokersAvailable:
                print(f"⚠️ Kafka not ready. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"❌ Kafka error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        print(f"❌ Could not connect to Kafka after {max_retries} attempts")
        self.kafka_ready = False
    
    def validate_texture_file(self, texture_file: UploadFile) -> bool:
        """Validate texture file format with parallel processing"""
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        file_extension = Path(texture_file.filename).suffix.lower()
        return file_extension in allowed_extensions
    
    def parallel_file_validation(self, ply_file: UploadFile, texture_file: Optional[UploadFile] = None) -> Dict[str, Any]:
        """Validate files in parallel using thread pools"""
        def validate_ply():
            if not ply_file.filename.endswith('.ply'):
                raise HTTPException(status_code=400, detail="Model file must be a PLY file")
            return True
        
        def validate_texture():
            if texture_file and not self.validate_texture_file(texture_file):
                raise HTTPException(
                    status_code=400, 
                    detail="Texture file must be in JPEG, JPG, or PNG format"
                )
            return texture_file is not None
        
        # Run validations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ply_future = executor.submit(validate_ply)
            texture_future = executor.submit(validate_texture)
            
            ply_valid = ply_future.result()
            has_texture = texture_future.result()
        
        return {"ply_valid": ply_valid, "has_texture": has_texture}
    
    def parallel_file_saving(self, job_id: str, object_name: str, 
                           ply_file: UploadFile, texture_file: Optional[UploadFile] = None) -> Dict[str, Any]:
        """Save files in parallel using optimized I/O operations"""
        job_dir = f"/app/uploads/{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        def save_ply_file():
            ply_path = f"{job_dir}/{object_name}.ply"
            with open(ply_path, "wb") as f:
                f.write(ply_file.file.read())
            return ply_path
        
        def save_texture_file():
            if texture_file:
                texture_extension = Path(texture_file.filename).suffix.lower()
                texture_path = f"{job_dir}/{object_name}_texture{texture_extension}"
                with open(texture_path, "wb") as f:
                    f.write(texture_file.file.read())
                return texture_path
            return None
        
        # Save files in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ply_future = executor.submit(save_ply_file)
            texture_future = executor.submit(save_texture_file)
            
            ply_path = ply_future.result()
            texture_path = texture_future.result()
        
        return {"ply_path": ply_path, "texture_path": texture_path}
    
    def parallel_image_preprocessing(self, texture_path: str) -> Dict[str, Any]:
        """Preprocess images in parallel for optimal performance"""
        if not texture_path or not os.path.exists(texture_path):
            return {"processed": False}
        
        def process_image():
            try:
                # Load and validate image
                image = Image.open(texture_path)
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Get image info
                width, height = image.size
                
                # Optional: Resize if too large (for memory efficiency)
                if width > 4096 or height > 4096:
                    image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
                    width, height = image.size
                
                # Save optimized image
                optimized_path = texture_path.replace('.', '_optimized.')
                image.save(optimized_path, quality=95, optimize=True)
                
                return {
                    "processed": True,
                    "original_size": (width, height),
                    "optimized_path": optimized_path
                }
            except Exception as e:
                print(f"Image preprocessing error: {e}")
                return {"processed": False, "error": str(e)}
        
        # Process image in dedicated thread
        future = self.image_executor.submit(process_image)
        return future.result()
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "name": gpu.name,
                    "utilization": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature
                })
        except:
            pass
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available / (1024**3),
            "gpu_info": gpu_info,
            "active_jobs": len(self.active_jobs)
        }
    
    @profile
    def process_object_high_performance(self, object_name: str, ply_file: UploadFile, 
                                      texture_file: Optional[UploadFile] = None) -> Dict[str, Any]:
        """Process object with high-performance parallel operations"""
        start_time = time.time()
        job_id = f"query_{object_name}_{int(time.time())}"
        
        try:
            # Parallel file validation
            validation_result = self.parallel_file_validation(ply_file, texture_file)
            
            # Parallel file saving
            file_paths = self.parallel_file_saving(job_id, object_name, ply_file, texture_file)
            
            # Parallel image preprocessing (if texture exists)
            image_processing = {}
            if file_paths["texture_path"]:
                image_processing = self.parallel_image_preprocessing(file_paths["texture_path"])
            
            # Create job record with resource usage
            processing_time = time.time() - start_time
            
            job = QueryProcessingJob(
                job_id=job_id,
                object_name=object_name,
                has_texture=validation_result["has_texture"],
                status="processing",
                created_at=start_time,
                processing_time=processing_time,
                resource_usage=self.get_system_performance_metrics()
            )
            
            self.active_jobs[job_id] = job
            
            # Prepare optimized Kafka message
            kafka_message = {
                'job_id': job_id,
                'object_name': object_name,
                'ply_path': file_paths["ply_path"],
                'texture_path': file_paths["texture_path"],
                'has_texture': validation_result["has_texture"],
                'processing_type': 'high_performance_query_object',
                'performance_metrics': {
                    'api_processing_time': processing_time,
                    'file_size_mb': os.path.getsize(file_paths["ply_path"]) / (1024**2),
                    'system_resources': self.resource_manager.resources.__dict__
                },
                'timestamp': time.time()
            }
            
            # Send to both CNOS renderer and point cloud generator simultaneously
            self.producer.send('cnos-rendering', kafka_message)
            self.producer.send('point-cloud-generation', kafka_message)
            
            # Memory cleanup
            gc.collect()
            
            return {
                "job_id": job_id,
                "object_name": object_name,
                "has_texture": validation_result["has_texture"],
                "status": "processing",
                "processing_time": processing_time,
                "parallel_operations": {
                    "file_validation": "completed",
                    "file_saving": "completed",
                    "image_preprocessing": image_processing.get("processed", False)
                },
                "resource_usage": job.resource_usage,
                "message": "High-performance parallel processing started successfully"
            }
            
        except Exception as e:
            logging.error(f"Error in high-performance processing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def setup_endpoints(self):
        """Setup FastAPI endpoints with high-performance features"""
        
        @self.app.post("/process_object")
        async def process_object(
            object_name: str = Form(...),
            ply_file: UploadFile = File(...),
            texture_file: Optional[UploadFile] = File(None),
            background_tasks: BackgroundTasks = BackgroundTasks()
        ):
            """Process PLY file with optional texture using high-performance parallel processing"""
            
            if not self.kafka_ready:
                raise HTTPException(status_code=503, detail="Kafka not available")
            
            # Process with high-performance parallel operations
            result = self.process_object_high_performance(object_name, ply_file, texture_file)
            
            # Add background task for cleanup
            background_tasks.add_task(self.cleanup_old_jobs)
            
            return JSONResponse(content=result)
        
        @self.app.get("/status/{job_id}")
        async def get_job_status(job_id: str):
            """Get processing status for a job"""
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "object_name": job.object_name,
                "has_texture": job.has_texture,
                "status": job.status,
                "created_at": job.created_at,
                "processing_time": job.processing_time,
                "resource_usage": job.resource_usage
            }
        
        @self.app.get("/performance")
        async def get_performance_metrics():
            """Get real-time system performance metrics"""
            return self.get_system_performance_metrics()
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check with system information"""
            return {
                "status": "healthy" if self.kafka_ready else "unhealthy",
                "kafka_ready": self.kafka_ready,
                "kafka_servers": self.kafka_servers,
                "system_resources": self.resource_manager.resources.__dict__,
                "active_jobs": len(self.active_jobs),
                "thread_pools": {
                    "file_workers": self.file_executor._max_workers,
                    "image_workers": self.image_executor._max_workers,
                    "io_workers": self.io_executor._max_workers
                },
                "service": "high-performance-query-api",
                "version": "2.0.0"
            }
        
        @self.app.get("/jobs")
        async def list_jobs():
            """List all active jobs with performance metrics"""
            return {
                "active_jobs": len(self.active_jobs),
                "jobs": [
                    {
                        "job_id": job_id,
                        "object_name": job.object_name,
                        "has_texture": job.has_texture,
                        "status": job.status,
                        "created_at": job.created_at,
                        "processing_time": job.processing_time
                    }
                    for job_id, job in self.active_jobs.items()
                ],
                "system_performance": self.get_system_performance_metrics()
            }
    
    def cleanup_old_jobs(self):
        """Clean up old job records to prevent memory leaks"""
        current_time = time.time()
        job_timeout = 3600  # 1 hour
        
        jobs_to_remove = [
            job_id for job_id, job in self.active_jobs.items()
            if current_time - job.created_at > job_timeout
        ]
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        if jobs_to_remove:
            print(f"🧹 Cleaned up {len(jobs_to_remove)} old job records")
    
    def run(self):
        """Start the high-performance FastAPI server"""
        print("🚀 Starting High-Performance Query API service...")
        print(f"📡 Listening on port 7861")
        print(f"🔗 Kafka servers: {self.kafka_servers}")
        print(f"💻 System Resources:")
        print(f"   - CPU cores: {self.resource_manager.resources.cpu_count}")
        print(f"   - Memory: {self.resource_manager.resources.memory_total:.1f}GB")
        print(f"   - GPUs: {self.resource_manager.resources.gpu_count}")
        print(f"🔧 Thread Pools:")
        print(f"   - File workers: {self.file_executor._max_workers}")
        print(f"   - Image workers: {self.image_executor._max_workers}")
        print(f"   - I/O workers: {self.io_executor._max_workers}")
        
        uvicorn.run(
            self.app, 
            host="0.0.0.0", 
            port=7861,
            workers=1,
            access_log=True,
            loop="asyncio"
        )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    api = HighPerformanceQueryAPI()
    api.run()

