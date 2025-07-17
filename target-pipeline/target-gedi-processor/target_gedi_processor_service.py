import os
import json
import time
import zipfile
import numpy as np
import torch
from typing import Dict, Any, List
from pathlib import Path
import open3d as o3d
from kafka import KafkaConsumer, KafkaProducer
from tqdm import tqdm
import tempfile
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
import sys

# Add GeDi and PointNet2 to path
sys.path.insert(0, '/app/gedi-repo')
sys.path.insert(0, '/app/pointnet2')

try:
    # Import GeDi modules
    from models.gedi import GeDi
    from utils.data_utils import load_point_cloud_gedi
    from utils.feature_utils import extract_dual_scale_features
    print("✅ GeDi modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import GeDi modules: {e}")


class TargetGeDiProcessor:
    """GeDi processor for target video sequences with dual-scale geometric features."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_gedi()
        self.setup_executors()
        self.system_info()
        
        print("🔷 Target GeDi Processor Service Ready")
    
    def system_info(self):
        """Display system information"""
        cpu_count = multiprocessing.cpu_count()
        memory_total = psutil.virtual_memory().total / (1024**3)
        
        print(f"🚀 System Resources:")
        print(f"   💻 CPU Cores: {cpu_count}")
        print(f"   📊 Memory: {memory_total:.1f}GB")
        print(f"   🎮 PyTorch version: {torch.__version__}")
        print(f"   🔧 Device: {self.device}")
        print(f"   🐍 Python version: {sys.version}")
    
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
    
    def setup_gedi(self):
        """Setup GeDi model for geometric feature extraction"""
        try:
            print("🔄 Loading GeDi model...")
            
            # Initialize GeDi model with dual-scale configuration
            self.gedi_model = GeDi(
                num_points=1000,  # Point cloud size from Point Cloud Generator
                num_features=64,  # Output feature dimension
                dual_scale=True,  # Enable dual-scale processing
                device=self.device
            )
            
            # Load pre-trained weights if available
            model_path = "/app/gedi-repo/pretrained/gedi_model.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.gedi_model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded pre-trained GeDi weights")
            else:
                print("⚠️ Using randomly initialized GeDi model")
            
            self.gedi_model.to(self.device)
            self.gedi_model.eval()
            
            print(f"✅ GeDi model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to setup GeDi model: {e}")
            # Fallback to basic geometric feature extraction
            self.gedi_model = None
    
    def setup_executors(self):
        """Setup optimized thread pools"""
        cpu_count = multiprocessing.cpu_count()
        self.io_executor = ThreadPoolExecutor(
            max_workers=min(cpu_count, 8),
            thread_name_prefix="gedi_worker"
        )
        print("✅ Thread executors initialized")
    
    def load_point_cloud(self, point_cloud_path: str) -> np.ndarray:
        """Load and preprocess point cloud for GeDi processing"""
        try:
            # Load point cloud using Open3D
            pcd = o3d.io.read_point_cloud(point_cloud_path)
            points = np.asarray(pcd.points)
            
            # Ensure we have exactly 1000 points
            if points.shape[0] != 1000:
                print(f"⚠️ Point cloud has {points.shape[0]} points, expected 1000")
                
                if points.shape[0] > 1000:
                    # Downsample to 1000 points
                    indices = np.random.choice(points.shape[0], 1000, replace=False)
                    points = points[indices]
                elif points.shape[0] < 1000:
                    # Upsample by repeating points
                    repeat_count = 1000 // points.shape[0]
                    remainder = 1000 % points.shape[0]
                    
                    repeated_points = np.tile(points, (repeat_count, 1))
                    if remainder > 0:
                        extra_points = points[:remainder]
                        points = np.vstack([repeated_points, extra_points])
                    else:
                        points = repeated_points
            
            # Normalize point cloud to unit sphere
            points_centered = points - np.mean(points, axis=0)
            max_dist = np.max(np.linalg.norm(points_centered, axis=1))
            if max_dist > 0:
                points_normalized = points_centered / max_dist
            else:
                points_normalized = points_centered
            
            return points_normalized
            
        except Exception as e:
            print(f"❌ Failed to load point cloud {point_cloud_path}: {e}")
            # Return random point cloud as fallback
            return np.random.randn(1000, 3).astype(np.float32)
    
    def extract_gedi_features(self, point_cloud: np.ndarray, frame_id: str) -> np.ndarray:
        """Extract geometric features using GeDi dual-scale processing"""
        try:
            # Convert to PyTorch tensor
            points_tensor = torch.from_numpy(point_cloud).float()
            points_tensor = points_tensor.unsqueeze(0)  # Add batch dimension
            points_tensor = points_tensor.transpose(2, 1)  # [B, 3, N]
            points_tensor = points_tensor.to(self.device)
            
            if self.gedi_model is not None:
                # Use GeDi model for feature extraction
                with torch.no_grad():
                    features = self.gedi_model(points_tensor)
                    
                    # Ensure output is 64-dimensional
                    if features.shape[-1] != 64:
                        # Project to 64 dimensions if needed
                        if not hasattr(self, 'feature_projection'):
                            self.feature_projection = torch.nn.Linear(
                                features.shape[-1], 64
                            ).to(self.device)
                        features = self.feature_projection(features)
                    
                    # L2 normalize features
                    features = torch.nn.functional.normalize(features, p=2, dim=-1)
                    
                    return features.cpu().numpy().flatten()
            else:
                # Fallback: Basic geometric feature extraction
                return self.extract_basic_geometric_features(point_cloud)
                
        except Exception as e:
            print(f"❌ GeDi feature extraction failed for frame {frame_id}: {e}")
            return self.extract_basic_geometric_features(point_cloud)
    
    def extract_basic_geometric_features(self, point_cloud: np.ndarray) -> np.ndarray:
        """Fallback basic geometric feature extraction"""
        try:
            features = []
            
            # Statistical features
            features.extend([
                np.mean(point_cloud, axis=0),  # Centroid (3D)
                np.std(point_cloud, axis=0),   # Standard deviation (3D)
                np.min(point_cloud, axis=0),   # Min coordinates (3D)
                np.max(point_cloud, axis=0),   # Max coordinates (3D)
            ])
            
            # Geometric properties
            centroid = np.mean(point_cloud, axis=0)
            distances = np.linalg.norm(point_cloud - centroid, axis=1)
            
            features.extend([
                np.mean(distances),    # Average distance from centroid
                np.std(distances),     # Distance standard deviation
                np.min(distances),     # Min distance
                np.max(distances),     # Max distance
            ])
            
            # Principal component analysis
            pca_matrix = np.cov(point_cloud.T)
            eigenvalues = np.linalg.eigvals(pca_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            features.extend(eigenvalues)  # PCA eigenvalues (3D)
            
            # Surface area approximation
            hull_volume = 0.0
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(point_cloud)
                hull_volume = hull.volume
            except:
                hull_volume = 0.0
            
            features.append(hull_volume)
            
            # Pad or truncate to exactly 64 dimensions
            features_array = np.array(features, dtype=np.float32)
            if len(features_array) < 64:
                # Pad with zeros
                padding = np.zeros(64 - len(features_array), dtype=np.float32)
                features_array = np.concatenate([features_array, padding])
            elif len(features_array) > 64:
                # Truncate to 64
                features_array = features_array[:64]
            
            # L2 normalize
            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm
            
            return features_array
            
        except Exception as e:
            print(f"❌ Basic geometric feature extraction failed: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def process_single_frame(self, frame_data: Dict[str, Any], temp_dirs: Dict[str, Path], 
                           output_dir: Path) -> str:
        """Process a single frame for GeDi geometric feature extraction"""
        
        try:
            frame_id = frame_data['frame_id']
            
            # Find corresponding point cloud file
            pc_filename = f"pointcloud_{frame_id}.ply"
            pc_path = temp_dirs['point_clouds'] / pc_filename
            
            if not pc_path.exists():
                raise FileNotFoundError(f"Point cloud not found: {pc_path}")
            
            # Load and preprocess point cloud
            point_cloud = self.load_point_cloud(str(pc_path))
            
            # Extract GeDi geometric features
            features = self.extract_gedi_features(point_cloud, frame_id)
            
            # Save features as numpy array
            feature_filename = f"gedi_features_{frame_id}.npy"
            feature_path = output_dir / feature_filename
            np.save(feature_path, features)
            
            print(f"✅ GeDi features extracted for frame {frame_id}: {features.shape}")
            return feature_filename
            
        except Exception as e:
            print(f"❌ Failed to process frame {frame_data.get('frame_id', 'unknown')}: {e}")
            return None
    
    def process_target_sequence(self, message: Dict[str, Any]):
        """Process target video sequence for GeDi geometric features"""
        
        try:
            job_id = message['job_id']
            sequence_name = message['sequence_name']
            point_clouds_zip = message['point_clouds_zip']
            frame_mapping = message['frame_mapping']
            
            print(f"🔷 Processing GeDi features for: {sequence_name}")
            print(f"📋 Total frames: {len(frame_mapping)}")
            
            # Create output directory
            output_dir = Path(f"/app/output/{job_id}/gedi_features")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract point clouds
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pc_dir = temp_path / "point_clouds"
                pc_dir.mkdir()
                
                print("📦 Extracting point clouds...")
                with zipfile.ZipFile(point_clouds_zip, 'r') as zip_ref:
                    zip_ref.extractall(pc_dir)
                
                temp_dirs = {'point_clouds': pc_dir}
                
                # Process frames in parallel
                feature_files = []
                
                print("🔷 Extracting GeDi geometric features...")
                
                max_workers = min(multiprocessing.cpu_count(), len(frame_mapping), 4)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_frame = {
                        executor.submit(
                            self.process_single_frame,
                            frame_data,
                            temp_dirs,
                            output_dir
                        ): frame_data
                        for frame_data in frame_mapping
                    }
                    
                    for future in tqdm(as_completed(future_to_frame),
                                     total=len(frame_mapping),
                                     desc="GeDi processing"):
                        result = future.result()
                        if result:
                            feature_files.append(result)
                
                # Create ZIP file of geometric features
                gedi_features_zip = output_dir.parent / "gedi_features.zip"
                with zipfile.ZipFile(gedi_features_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                    for feature_file in feature_files:
                        feature_path = output_dir / feature_file
                        if feature_path.exists():
                            zip_ref.write(feature_path, feature_file)
                
                print(f"✅ Generated {len(feature_files)} GeDi feature files")
                print(f"💾 GeDi features ZIP: {gedi_features_zip}")
                
                # Send completion message
                completion_message = {
                    'job_id': job_id,
                    'sequence_name': sequence_name,
                    'gedi_features_zip': str(gedi_features_zip),
                    'total_frames': len(feature_files),
                    'processing_type': 'target_gedi_complete',
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
    
    def run(self):
        """Run the target GeDi processor service"""
        print("🚀 Starting Target GeDi Processor Service")
        print("🔄 Waiting for point cloud processing requests...")
        
        try:
            for message in self.consumer:
                self.process_target_sequence(message.value)
                
        except KeyboardInterrupt:
            print("🛑 Shutting down Target GeDi Processor")
        except Exception as e:
            print(f"❌ Service error: {e}")
        finally:
            self.io_executor.shutdown(wait=True)

if __name__ == "__main__":
    processor = TargetGeDiProcessor()
    processor.run()
