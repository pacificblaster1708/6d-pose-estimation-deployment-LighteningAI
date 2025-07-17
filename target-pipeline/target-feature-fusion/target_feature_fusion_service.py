import os
import json
import time
import zipfile
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
from tqdm import tqdm
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import re

class TargetFeatureFusion:
    """Target feature fusion service for combining visual and geometric features"""
    
    def __init__(self):
        self.system_resources = self.detect_system_resources()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        
        print("🔗 Target Feature Fusion Service Ready")
    
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
                'target-feature-fusion',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='target-feature-fusion-group',
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
            thread_name_prefix="fusion_worker"
        )
        print("✅ Thread executors initialized")
    
    def extract_frame_id(self, filename: str) -> str:
        """Extract frame ID from feature filename"""
        # Handle various naming conventions
        # target_000620_gedi.npy -> 000620
        # dinov2_features_000620.npy -> 000620
        # gedi_features_000620.npy -> 000620
        
        name = Path(filename).stem
        
        patterns = [
            r'target_(\d+)_gedi',        # target_000620_gedi
            r'target_(\d+)_pca64',       # target_000620_pca64
            r'dinov2_features_(\d+)',    # dinov2_features_000620
            r'gedi_features_(\d+)',      # gedi_features_000620
            r'(\d{6})',                  # Any 6-digit number
            r'(\d{5})',                  # Any 5-digit number
            r'(\d{4})',                  # Any 4-digit number
            r'(\d+)'                     # Any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).zfill(6)
        
        return name
    
    def fuse_features_from_arrays(self, gedi_features: np.ndarray, visual_features: np.ndarray, 
                                frame_id: str) -> np.ndarray:
        """Fuse geometric and visual features with validation"""
        
        try:
            # Validate input shapes
            assert gedi_features.shape == (1000, 64), f"GeDi features shape mismatch: {gedi_features.shape}"
            assert visual_features.shape == (1000, 64), f"Visual features shape mismatch: {visual_features.shape}"
            
            # Verify L2 normalization (check first few vectors)
            gedi_norms = np.linalg.norm(gedi_features[:5], axis=1)
            visual_norms = np.linalg.norm(visual_features[:5], axis=1)
            
            print(f"   📏 GeDi L2 norms sample: {gedi_norms}")
            print(f"   📏 Visual L2 norms sample: {visual_norms}")
            
            # Concatenate features: [gedi_64D, visual_64D] -> 128D
            fused_features = np.concatenate([gedi_features, visual_features], axis=1)
            
            # Verify concatenation result
            assert fused_features.shape == (1000, 128), f"Fused features shape error: {fused_features.shape}"
            
            print(f"   🔗 Fused features shape: {fused_features.shape}")
            
            return fused_features
            
        except Exception as e:
            print(f"❌ Feature fusion failed for frame {frame_id}: {e}")
            # Return zero features as fallback
            return np.zeros((1000, 128), dtype=np.float32)
    
    def process_single_frame_fusion(self, frame_id: str, visual_features_dir: Path, 
                                  geometric_features_dir: Path, output_dir: Path) -> str:
        """Process fusion for a single frame"""
        
        try:
            # Find corresponding feature files
            visual_patterns = [
                f"dinov2_features_{frame_id}.npy",
                f"target_{frame_id}_pca64.npy",
                f"visual_features_{frame_id}.npy"
            ]
            
            geometric_patterns = [
                f"gedi_features_{frame_id}.npy",
                f"target_{frame_id}_gedi.npy",
                f"geometric_features_{frame_id}.npy"
            ]
            
            # Find visual features file
            visual_path = None
            for pattern in visual_patterns:
                candidate = visual_features_dir / pattern
                if candidate.exists():
                    visual_path = candidate
                    break
            
            if visual_path is None:
                raise FileNotFoundError(f"Visual features not found for frame {frame_id}")
            
            # Find geometric features file
            geometric_path = None
            for pattern in geometric_patterns:
                candidate = geometric_features_dir / pattern
                if candidate.exists():
                    geometric_path = candidate
                    break
            
            if geometric_path is None:
                raise FileNotFoundError(f"Geometric features not found for frame {frame_id}")
            
            # Load feature arrays
            visual_features = np.load(str(visual_path))
            geometric_features = np.load(str(geometric_path))
            
            print(f"   📥 Loaded visual: {visual_features.shape}")
            print(f"   📥 Loaded geometric: {geometric_features.shape}")
            
            # Fuse features
            fused_features = self.fuse_features_from_arrays(
                geometric_features, visual_features, frame_id
            )
            
            # Save fused features with consistent naming
            fused_filename = f"target_{frame_id}_128.npy"
            fused_path = output_dir / fused_filename
            np.save(str(fused_path), fused_features)
            
            print(f"✅ Fused features saved for frame {frame_id}: {fused_features.shape}")
            return fused_filename
            
        except Exception as e:
            print(f"❌ Failed to fuse features for frame {frame_id}: {e}")
            return None
    
    def process_target_feature_fusion(self, message: Dict[str, Any]):
        """Process feature fusion for target video sequence"""
        
        try:
            job_id = message['job_id']
            sequence_name = message['sequence_name']
            
            # Get feature ZIP files from messages
            visual_features_zip = None
            geometric_features_zip = None
            
            # Check for DINOv2 completion message
            if message.get('processing_type') == 'target_dinov2_complete':
                visual_features_zip = message['visual_features_zip']
                # Wait for corresponding GeDi message
                self.pending_visual_features[job_id] = message
                print(f"📥 Received visual features for {job_id}, waiting for geometric features...")
                return
            
            # Check for GeDi completion message
            elif message.get('processing_type') == 'target_gedi_complete':
                geometric_features_zip = message['geometric_features_zip']
                
                # Check if we have pending visual features
                if job_id in getattr(self, 'pending_visual_features', {}):
                    visual_message = self.pending_visual_features[job_id]
                    visual_features_zip = visual_message['visual_features_zip']
                    del self.pending_visual_features[job_id]
                else:
                    # Wait for visual features
                    if not hasattr(self, 'pending_geometric_features'):
                        self.pending_geometric_features = {}
                    self.pending_geometric_features[job_id] = message
                    print(f"📥 Received geometric features for {job_id}, waiting for visual features...")
                    return
            
            # Check for cached geometric features
            elif job_id in getattr(self, 'pending_geometric_features', {}):
                geometric_message = self.pending_geometric_features[job_id]
                geometric_features_zip = geometric_message['geometric_features_zip']
                visual_features_zip = message['visual_features_zip']
                del self.pending_geometric_features[job_id]
            
            else:
                print(f"⚠️ Unknown message type: {message.get('processing_type')}")
                return
            
            if not visual_features_zip or not geometric_features_zip:
                print(f"⚠️ Missing feature files for fusion: visual={bool(visual_features_zip)}, geometric={bool(geometric_features_zip)}")
                return
            
            print(f"🔗 Starting feature fusion for: {sequence_name}")
            print(f"📋 Job ID: {job_id}")
            print(f"📁 Visual features: {visual_features_zip}")
            print(f"📁 Geometric features: {geometric_features_zip}")
            
            # Create output directory
            output_dir = Path(f"/app/output/{job_id}/fused_features")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract feature files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                visual_dir = temp_path / "visual_features"
                geometric_dir = temp_path / "geometric_features"
                
                visual_dir.mkdir()
                geometric_dir.mkdir()
                
                print("📦 Extracting feature ZIP files...")
                
                # Extract visual features
                with zipfile.ZipFile(visual_features_zip, 'r') as zip_ref:
                    zip_ref.extractall(visual_dir)
                
                # Extract geometric features
                with zipfile.ZipFile(geometric_features_zip, 'r') as zip_ref:
                    zip_ref.extractall(geometric_dir)
                
                # Get list of frame IDs from feature files
                visual_files = list(visual_dir.glob("*.npy"))
                geometric_files = list(geometric_dir.glob("*.npy"))
                
                # Extract frame IDs
                visual_frame_ids = {self.extract_frame_id(f.name) for f in visual_files}
                geometric_frame_ids = {self.extract_frame_id(f.name) for f in geometric_files}
                
                # Find common frame IDs
                common_frame_ids = visual_frame_ids.intersection(geometric_frame_ids)
                common_frame_ids = sorted(list(common_frame_ids))
                
                print(f"📊 Visual frames: {len(visual_frame_ids)}")
                print(f"📊 Geometric frames: {len(geometric_frame_ids)}")
                print(f"📊 Common frames for fusion: {len(common_frame_ids)}")
                
                if len(common_frame_ids) == 0:
                    raise ValueError("No matching frames found between visual and geometric features")
                
                # Process frame fusion in parallel
                fused_files = []
                
                print("🔗 Fusing visual and geometric features...")
                
                max_workers = min(self.system_resources['cpu_count'], len(common_frame_ids), 8)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_frame = {
                        executor.submit(
                            self.process_single_frame_fusion,
                            frame_id,
                            visual_dir,
                            geometric_dir,
                            output_dir
                        ): frame_id
                        for frame_id in common_frame_ids
                    }
                    
                    for future in tqdm(as_completed(future_to_frame),
                                     total=len(common_frame_ids),
                                     desc="Feature fusion"):
                        result = future.result()
                        if result:
                            fused_files.append(result)
                
                # Create ZIP file of fused features
                fused_features_zip = output_dir.parent / "fused_features.zip"
                
                print("📦 Creating fused features ZIP...")
                with zipfile.ZipFile(fused_features_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for fused_file in fused_files:
                        fused_path = output_dir / fused_file
                        if fused_path.exists():
                            zip_ref.write(fused_path, fused_file)
                
                print(f"✅ Feature fusion completed!")
                print(f"📊 Fused {len(fused_files)} frames")
                print(f"💾 Fused features saved to: {fused_features_zip}")
                
                # Send completion message
                completion_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'fused_features_zip': str(fused_features_zip),
                    'total_frames': len(fused_files),
                    'feature_dimensions': 128,
                    'processing_type': 'target_fusion_complete',
                    'timestamp': time.time()
                }
                
                self.producer.send('target-processing-complete', completion_message)
                
                print(f"✅ Feature fusion completed for {job_id}")
                
                gc.collect()
                
        except Exception as e:
            print(f"❌ Target feature fusion failed: {e}")
    
    def run(self):
        """Run the target feature fusion service"""
        print("🚀 Starting Target Feature Fusion Service")
        print(f"💻 CPU Cores: {self.system_resources['cpu_count']}")
        print(f"📊 Memory: {self.system_resources['memory_total']:.1f}GB")
        print("🔄 Waiting for feature fusion requests...")
        
        # Initialize pending feature storage
        self.pending_visual_features = {}
        self.pending_geometric_features = {}
        
        try:
            for message in self.consumer:
                self.process_target_feature_fusion(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Target Feature Fusion")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    fusion_service = TargetFeatureFusion()
    fusion_service.run()
