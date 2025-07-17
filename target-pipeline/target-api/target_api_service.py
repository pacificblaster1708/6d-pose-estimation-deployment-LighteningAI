import os
import json
import time
import zipfile
import asyncio
import aiofiles
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import psutil
import subprocess

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from kafka import KafkaProducer
import uuid
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import tempfile
import gc

# Set uvloop for better async performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

class SystemResourceManager:
    """Manage system resources for optimal performance"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024**3)
        self.gpu_count = self.detect_gpus()
        self.cuda_available = self.detect_cuda()
        
        print(f"🚀 System Resources Detected:")
        print(f"   💻 CPU Cores: {self.cpu_count}")
        print(f"   📊 Memory: {self.memory_total:.1f}GB")
        print(f"   🎮 GPU Count: {self.gpu_count}")
        print(f"   🔧 CUDA Available: {self.cuda_available}")
    
    def detect_gpus(self) -> int:
        """Detect available GPUs without external dependencies"""
        try:
            # Try nvidia-smi command
            result = subprocess.run(['nvidia-smi', '-L'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return len([line for line in result.stdout.split('\n') 
                          if 'GPU' in line and line.strip()])
        except:
            pass
        
        try:
            # Try PyTorch CUDA detection
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except:
            pass
        
        return 0
    
    def detect_cuda(self) -> bool:
        """Detect CUDA availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads"""
        return min(self.cpu_count * 2, 32)

class HighPerformanceTargetAPI:
    """High-performance Target API with multithreading and async support"""
    
    def __init__(self):
        self.system_resources = SystemResourceManager()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        self.active_jobs = {}
        
        print("🎬 High-Performance Target API Ready")
    
    def setup_kafka(self):
        """Setup high-throughput Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                batch_size=32768,
                linger_ms=10,
                compression_type='lz4',
                acks='all',
                retries=3
            )
            print("✅ High-throughput Kafka producer ready")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
            self.producer = None
    
    def setup_executors(self):
        """Setup optimized thread and process pools"""
        self.io_executor = ThreadPoolExecutor(
            max_workers=self.system_resources.get_optimal_workers(),
            thread_name_prefix="target_api_io"
        )
        
        self.process_executor = ProcessPoolExecutor(
            max_workers=min(self.system_resources.cpu_count, 8)
        )
        
        print(f"✅ Executors initialized: {self.system_resources.get_optimal_workers()} I/O workers")
    
    async def validate_video_sequence_files(self, rgb_frames: UploadFile, binary_masks: UploadFile,
                                          depth_images: UploadFile, camera_intrinsics: UploadFile) -> Dict[str, Any]:
        """Validate uploaded video sequence files with multithreading"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'file_info': {},
            'frame_analysis': {}
        }
        
        try:
            async def validate_file(file: UploadFile, file_type: str):
                """Validate individual file"""
                file_info = {
                    'filename': file.filename,
                    'size': 0,
                    'type': file_type,
                    'content_type': file.content_type
                }
                
                # Read file content for validation
                content = await file.read()
                file_info['size'] = len(content)
                
                # Reset file pointer
                await file.seek(0)
                
                # Validate file type and content
                if file_type in ['rgb_frames', 'binary_masks', 'depth_images']:
                    if not file.filename.lower().endswith('.zip'):
                        validation_results['errors'].append(f"{file_type} must be a ZIP file")
                        validation_results['valid'] = False
                    
                    # Quick ZIP validation
                    try:
                        import io
                        zip_buffer = io.BytesIO(content)
                        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                            file_count = len([f for f in zip_ref.namelist() 
                                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                            file_info['frame_count'] = file_count
                    except zipfile.BadZipFile:
                        validation_results['errors'].append(f"{file_type} is not a valid ZIP file")
                        validation_results['valid'] = False
                
                elif file_type == 'camera_intrinsics':
                    if not file.filename.lower().endswith('.json'):
                        validation_results['errors'].append("Camera intrinsics must be a JSON file")
                        validation_results['valid'] = False
                    
                    # Validate JSON structure
                    try:
                        intrinsics_data = json.loads(content.decode('utf-8'))
                        file_info['intrinsics_keys'] = list(intrinsics_data.keys())
                    except json.JSONDecodeError:
                        validation_results['errors'].append("Camera intrinsics is not valid JSON")
                        validation_results['valid'] = False
                
                return file_type, file_info
            
            # Run file validations in parallel
            validation_tasks = [
                validate_file(rgb_frames, 'rgb_frames'),
                validate_file(binary_masks, 'binary_masks'),
                validate_file(depth_images, 'depth_images'),
                validate_file(camera_intrinsics, 'camera_intrinsics')
            ]
            
            results = await asyncio.gather(*validation_tasks)
            
            # Collect validation results
            for file_type, file_info in results:
                validation_results['file_info'][file_type] = file_info
            
            # Cross-validate frame counts
            frame_counts = [
                validation_results['file_info'].get('rgb_frames', {}).get('frame_count', 0),
                validation_results['file_info'].get('binary_masks', {}).get('frame_count', 0),
                validation_results['file_info'].get('depth_images', {}).get('frame_count', 0)
            ]
            
            if len(set(frame_counts)) > 1:
                validation_results['errors'].append(
                    f"Frame count mismatch: RGB={frame_counts[0]}, Masks={frame_counts[1]}, Depth={frame_counts[2]}"
                )
                validation_results['valid'] = False
            else:
                validation_results['frame_analysis']['total_frames'] = frame_counts[0]
            
            return validation_results
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            return validation_results
    
    async def save_uploaded_files_parallel(self, job_id: str, files: Dict[str, UploadFile]) -> Dict[str, str]:
        """Save uploaded files in parallel for maximum I/O performance"""
        
        job_dir = Path(f"/app/data/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        async def save_file(file_type: str, file: UploadFile):
            """Save individual file"""
            try:
                file_extension = Path(file.filename).suffix
                save_path = job_dir / f"{file_type}{file_extension}"
                
                async with aiofiles.open(save_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                
                return file_type, str(save_path)
            except Exception as e:
                print(f"❌ Failed to save {file_type}: {e}")
                return file_type, None
        
        # Save all files in parallel
        save_tasks = [
            save_file(file_type, file) for file_type, file in files.items()
        ]
        
        results = await asyncio.gather(*save_tasks)
        
        # Collect file paths
        for file_type, file_path in results:
            if file_path:
                file_paths[file_type] = file_path
        
        return file_paths
    
    def create_frame_correspondence_mapping(self, file_paths: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create frame correspondence mapping with multithreading"""
        
        def extract_frame_list(zip_path: str) -> List[str]:
            """Extract frame list from ZIP file"""
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                return sorted([f for f in zip_ref.namelist() 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                             and not f.startswith('__MACOSX')])
        
        def extract_frame_id(filename: str) -> str:
            """Extract frame ID from filename"""
            import re
            name = Path(filename).stem
            match = re.search(r'(\d+)', name)
            return match.group(1).zfill(6) if match else name
        
        # Extract frame lists in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            rgb_future = executor.submit(extract_frame_list, file_paths['rgb_frames'])
            mask_future = executor.submit(extract_frame_list, file_paths['binary_masks'])
            depth_future = executor.submit(extract_frame_list, file_paths['depth_images'])
            
            rgb_frames = rgb_future.result()
            mask_frames = mask_future.result()
            depth_frames = depth_future.result()
        
        # Load camera intrinsics
        with open(file_paths['camera_intrinsics'], 'r') as f:
            camera_intrinsics = json.load(f)
        
        # Create frame correspondence mapping
        frame_mapping = []
        
        for rgb_frame in rgb_frames:
            frame_id = extract_frame_id(rgb_frame)
            
            # Find corresponding files
            corresponding_mask = next((f for f in mask_frames if frame_id in f), None)
            corresponding_depth = next((f for f in depth_frames if frame_id in f), None)
            
            if corresponding_mask and corresponding_depth:
                # Get frame-specific camera intrinsics
                intrinsics = camera_intrinsics.get(frame_id, 
                           camera_intrinsics.get('default',
                           list(camera_intrinsics.values())[0] if camera_intrinsics else {}))
                
                frame_mapping.append({
                    'frame_id': frame_id,
                    'rgb_frame': rgb_frame,
                    'mask_frame': corresponding_mask,
                    'depth_frame': corresponding_depth,
                    'camera_intrinsics': intrinsics
                })
        
        # Sort by frame ID for ascending order processing
        frame_mapping.sort(key=lambda x: int(x['frame_id']))
        
        return frame_mapping
    
    async def process_target_sequence_async(self, job_id: str, sequence_name: str,
                                          file_paths: Dict[str, str], frame_mapping: List[Dict[str, Any]]):
        """Asynchronously process target sequence with multithreading"""
        
        try:
            print(f"🎬 Processing target sequence: {sequence_name}")
            print(f"📋 Job ID: {job_id}")
            print(f"📊 Total frames: {len(frame_mapping)}")
            
            # Update job status
            self.active_jobs[job_id] = {
                'sequence_name': sequence_name,
                'status': 'processing',
                'total_frames': len(frame_mapping),
                'processed_frames': 0,
                'start_time': time.time()
            }
            
            # Send processing message to Kafka
            if self.producer:
                processing_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'file_paths': file_paths,
                    'frame_mapping': frame_mapping,
                    'processing_type': 'target_sequence_processing',
                    'performance_info': {
                        'cpu_cores': self.system_resources.cpu_count,
                        'gpu_count': self.system_resources.gpu_count,
                        'cuda_available': self.system_resources.cuda_available
                    },
                    'timestamp': time.time()
                }
                
                self.producer.send('target-point-cloud-generation', processing_message)
                print(f"✅ Processing message sent to Kafka")
            
            # Update job status
            self.active_jobs[job_id]['status'] = 'sent_to_pipeline'
            
        except Exception as e:
            print(f"❌ Failed to process target sequence: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'failed'
                self.active_jobs[job_id]['error'] = str(e)

# Initialize the high-performance API
target_api = HighPerformanceTargetAPI()

# Create FastAPI app
app = FastAPI(
    title="High-Performance Target Pipeline API",
    description="GPU-accelerated target video sequence processing for 6D pose estimation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_target_sequence")
async def process_target_sequence(
    background_tasks: BackgroundTasks,
    sequence_name: str = Form(...),
    rgb_frames: UploadFile = File(...),
    binary_masks: UploadFile = File(...),
    depth_images: UploadFile = File(...),
    camera_intrinsics: UploadFile = File(...)
):
    """Process target video sequence with high-performance multithreading"""
    
    try:
        # Generate unique job ID
        job_id = f"target_{sequence_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        print(f"🎬 Received target sequence: {sequence_name}")
        print(f"📋 Job ID: {job_id}")
        
        # Validate uploaded files in parallel
        validation_results = await target_api.validate_video_sequence_files(
            rgb_frames, binary_masks, depth_images, camera_intrinsics
        )
        
        if not validation_results['valid']:
            raise HTTPException(status_code=400, detail={
                'message': 'File validation failed',
                'errors': validation_results['errors']
            })
        
        print(f"✅ File validation passed: {validation_results['frame_analysis']['total_frames']} frames")
        
        # Save uploaded files in parallel
        files = {
            'rgb_frames': rgb_frames,
            'binary_masks': binary_masks,
            'depth_images': depth_images,
            'camera_intrinsics': camera_intrinsics
        }
        
        file_paths = await target_api.save_uploaded_files_parallel(job_id, files)
        print(f"✅ Files saved: {list(file_paths.keys())}")
        
        # Create frame correspondence mapping
        frame_mapping = target_api.create_frame_correspondence_mapping(file_paths)
        print(f"✅ Frame correspondence created: {len(frame_mapping)} frames mapped")
        
        # Process sequence asynchronously
        background_tasks.add_task(
            target_api.process_target_sequence_async,
            job_id, sequence_name, file_paths, frame_mapping
        )
        
        return JSONResponse({
            'status': 'accepted',
            'job_id': job_id,
            'sequence_name': sequence_name,
            'file_validation': validation_results,
            'processing_info': {
                'total_frames': len(frame_mapping),
                'cpu_cores_utilized': target_api.system_resources.cpu_count,
                'gpu_acceleration': target_api.system_resources.cuda_available,
                'multithreading_enabled': True
            },
            'message': 'Target sequence processing initiated with high-performance multithreading'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get job processing status"""
    if job_id not in target_api.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = target_api.active_jobs[job_id]
    
    # Calculate processing time
    if 'start_time' in job_info:
        job_info['processing_time'] = time.time() - job_info['start_time']
    
    return JSONResponse(job_info)

@app.get("/health")
async def health_check():
    """Health check with system resource information"""
    return JSONResponse({
        'status': 'healthy',
        'service': 'target-api',
        'version': '2.0.0',
        'performance_features': {
            'multithreading': True,
            'gpu_acceleration': target_api.system_resources.cuda_available,
            'parallel_file_processing': True,
            'async_processing': True
        },
        'system_resources': {
            'cpu_cores': target_api.system_resources.cpu_count,
            'memory_gb': round(target_api.system_resources.memory_total, 1),
            'gpu_count': target_api.system_resources.gpu_count
        }
    })

@app.get("/")
async def root():
    """API root with performance information"""
    return JSONResponse({
        'message': 'High-Performance Target Pipeline API',
        'version': '2.0.0',
        'features': [
            'Multithreaded file processing',
            'GPU acceleration support',
            'Parallel video sequence validation',
            'Async background processing',
            'High-throughput Kafka integration'
        ],
        'endpoints': {
            'process_sequence': '/process_target_sequence',
            'job_status': '/job_status/{job_id}',
            'health': '/health'
        }
    })

if __name__ == "__main__":
    uvicorn.run(
        "target_api_service:app",
        host="0.0.0.0",
        port=8001,
        workers=1,
        reload=False,
        access_log=True
    )
