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
import cv2
import open3d as o3d
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import faiss
import matplotlib.pyplot as plt
import transforms3d
from flask import Flask, request, jsonify, send_file

class PoseOverlayPipeline:
    """Pose overlay pipeline for 6D pose estimation and visualization"""
    
    def __init__(self):
        self.system_resources = self.detect_system_resources()
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'target-kafka:29092')
        self.setup_kafka()
        self.setup_executors()
        self.setup_flask()
        
        # Initialize FAISS index for efficient similarity search
        self.faiss_index = None
        self.target_features = None
        self.target_metadata = None
        
        print("🎯 Pose Overlay Pipeline Service Ready")
    
    def detect_system_resources(self):
        """Detect system resources"""
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
                'pose-overlay-processing',
                bootstrap_servers=[self.kafka_servers],
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='pose-overlay-group'
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_servers],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("✅ Kafka connected")
        except Exception as e:
            print(f"❌ Kafka setup failed: {e}")
    
    def setup_executors(self):
        """Setup thread executors"""
        self.executor = ThreadPoolExecutor(
            max_workers=min(self.system_resources['cpu_count'], 8),
            thread_name_prefix="pose_worker"
        )
        print("✅ Thread executors initialized")
    
    def setup_flask(self):
        """Setup Flask web server for API endpoints"""
        self.app = Flask(__name__)
        self.setup_routes()
        print("✅ Flask server initialized")
    
    def setup_routes(self):
        """Setup API routes"""
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'service': 'pose-overlay-pipeline'})
        
        @self.app.route('/process_pose_estimation', methods=['POST'])
        def process_pose_estimation():
            return self.handle_pose_estimation_request()
        
        @self.app.route('/load_target_features', methods=['POST'])
        def load_target_features():
            return self.handle_target_features_loading()
    
    def load_target_features_from_zip(self, target_features_zip: str) -> bool:
        """Load target features from ZIP file and build FAISS index"""
        try:
            print(f"📦 Loading target features from: {target_features_zip}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract target features
                with zipfile.ZipFile(target_features_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Load all feature files
                feature_files = list(temp_path.glob("*.npy"))
                
                if not feature_files:
                    raise ValueError("No feature files found in target ZIP")
                
                features_list = []
                metadata_list = []
                
                for feature_file in tqdm(feature_files, desc="Loading target features"):
                    features = np.load(feature_file)
                    
                    # Validate feature dimensions
                    if features.shape[1] != 128:
                        print(f"⚠️ Invalid feature dimensions: {features.shape}")
                        continue
                    
                    frame_id = feature_file.stem
                    
                    # Store features and metadata
                    features_list.append(features)
                    metadata_list.extend([{
                        'frame_id': frame_id,
                        'point_index': i,
                        'file_path': str(feature_file)
                    } for i in range(features.shape[0])])
                
                # Concatenate all features
                self.target_features = np.vstack(features_list).astype(np.float32)
                self.target_metadata = metadata_list
                
                # Build FAISS index for efficient similarity search
                self.build_faiss_index()
                
                print(f"✅ Loaded {len(self.target_features)} target feature vectors")
                return True
                
        except Exception as e:
            print(f"❌ Failed to load target features: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        try:
            print("🔍 Building FAISS index for similarity search...")
            
            # Normalize features for cosine similarity
            normalized_features = self.target_features / np.linalg.norm(
                self.target_features, axis=1, keepdims=True
            )
            
            # Create FAISS index
            dimension = normalized_features.shape[1]  # 128
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Add features to index
            self.faiss_index.add(normalized_features)
            
            print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            print(f"❌ Failed to build FAISS index: {e}")
    
    def find_best_matches(self, query_features: np.ndarray, k: int = 10) -> List[Dict]:
        """Find best matches for query features using FAISS"""
        try:
            # Normalize query features
            query_normalized = query_features / np.linalg.norm(
                query_features, axis=1, keepdims=True
            )
            
            # Search for similar features
            similarities, indices = self.faiss_index.search(query_normalized, k)
            
            matches = []
            for i, (sim_scores, idx_list) in enumerate(zip(similarities, indices)):
                point_matches = []
                for j, (similarity, target_idx) in enumerate(zip(sim_scores, idx_list)):
                    if target_idx < len(self.target_metadata):
                        point_matches.append({
                            'similarity': float(similarity),
                            'target_metadata': self.target_metadata[target_idx],
                            'rank': j + 1
                        })
                
                matches.append({
                    'query_point_index': i,
                    'matches': point_matches
                })
            
            return matches
            
        except Exception as e:
            print(f"❌ Feature matching failed: {e}")
            return []
    
    def estimate_pose_from_matches(self, matches: List[Dict], 
                                 query_point_cloud: np.ndarray) -> Dict:
        """Estimate 6D pose from feature matches"""
        try:
            # Extract corresponding points
            query_points = []
            target_points = []
            
            for match in matches:
                if match['matches']:
                    best_match = match['matches'][0]  # Top match
                    
                    query_idx = match['query_point_index']
                    target_metadata = best_match['target_metadata']
                    
                    # Add points if similarity is above threshold
                    if best_match['similarity'] > 0.7:  # Similarity threshold
                        query_points.append(query_point_cloud[query_idx])
                        
                        # For demo, use frame_id as approximate target point
                        # In production, you'd load actual target point cloud
                        target_points.append(query_point_cloud[query_idx] + np.random.normal(0, 0.1, 3))
            
            if len(query_points) < 4:
                raise ValueError("Insufficient matches for pose estimation")
            
            query_points = np.array(query_points)
            target_points = np.array(target_points)
            
            # Estimate pose using ICP-like method
            pose_result = self.compute_rigid_transformation(query_points, target_points)
            
            return {
                'rotation_matrix': pose_result['rotation'].tolist(),
                'translation_vector': pose_result['translation'].tolist(),
                'num_matches': len(query_points),
                'average_similarity': np.mean([m['matches'][0]['similarity'] for m in matches if m['matches']]),
                'pose_confidence': min(1.0, len(query_points) / 100.0)
            }
            
        except Exception as e:
            print(f"❌ Pose estimation failed: {e}")
            return None
    
    def compute_rigid_transformation(self, source: np.ndarray, 
                                   target: np.ndarray) -> Dict:
        """Compute rigid transformation between point sets"""
        try:
            # Center the point sets
            source_centroid = np.mean(source, axis=0)
            target_centroid = np.mean(target, axis=0)
            
            source_centered = source - source_centroid
            target_centered = target - target_centroid
            
            # Compute cross-covariance matrix
            H = source_centered.T @ target_centered
            
            # SVD decomposition
            U, S, Vt = np.linalg.svd(H)
            
            # Compute rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation matrix
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = target_centroid - R @ source_centroid
            
            return {
                'rotation': R,
                'translation': t,
                'error': np.mean(np.linalg.norm(target - (R @ source.T + t.reshape(-1, 1)).T, axis=1))
            }
            
        except Exception as e:
            print(f"❌ Rigid transformation failed: {e}")
            return {'rotation': np.eye(3), 'translation': np.zeros(3), 'error': float('inf')}
    
    def create_pose_overlay(self, pose_result: Dict, query_image: np.ndarray = None) -> str:
        """Create pose overlay visualization"""
        try:
            output_dir = Path("/app/output/pose_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Rotation matrix visualization
            rotation_matrix = np.array(pose_result['rotation_matrix'])
            im1 = axes[0, 0].imshow(rotation_matrix, cmap='viridis')
            axes[0, 0].set_title('Estimated Rotation Matrix')
            axes[0, 0].set_xlabel('Column')
            axes[0, 0].set_ylabel('Row')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot 2: Translation vector
            translation = np.array(pose_result['translation_vector'])
            axes[0, 1].bar(['X', 'Y', 'Z'], translation)
            axes[0, 1].set_title('Translation Vector')
            axes[0, 1].set_ylabel('Translation (units)')
            
            # Plot 3: Pose confidence metrics
            metrics = ['Matches', 'Similarity', 'Confidence']
            values = [
                pose_result['num_matches'],
                pose_result['average_similarity'],
                pose_result['pose_confidence']
            ]
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Pose Estimation Metrics')
            axes[1, 0].set_ylabel('Value')
            
            # Plot 4: Summary text
            axes[1, 1].text(0.1, 0.8, f"Matches: {pose_result['num_matches']}", fontsize=12)
            axes[1, 1].text(0.1, 0.6, f"Avg Similarity: {pose_result['average_similarity']:.3f}", fontsize=12)
            axes[1, 1].text(0.1, 0.4, f"Confidence: {pose_result['pose_confidence']:.3f}", fontsize=12)
            axes[1, 1].set_title('Pose Estimation Summary')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            timestamp = int(time.time())
            output_path = output_dir / f"pose_overlay_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Pose overlay created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Pose overlay creation failed: {e}")
            return None
    
    def handle_pose_estimation_request(self):
        """Handle pose estimation API request"""
        try:
            # Get uploaded files
            if 'query_features' not in request.files:
                return jsonify({'error': 'Query features file required'}), 400
            
            query_features_file = request.files['query_features']
            
            # Load query features
            query_features = np.load(query_features_file)
            
            # Validate dimensions
            if query_features.shape[1] != 128:
                return jsonify({'error': f'Invalid feature dimensions: {query_features.shape}'}), 400
            
            # Find matches
            matches = self.find_best_matches(query_features)
            
            if not matches:
                return jsonify({'error': 'No matches found'}), 404
            
            # Estimate pose
            # For demo, create dummy point cloud
            query_point_cloud = np.random.randn(query_features.shape[0], 3)
            
            pose_result = self.estimate_pose_from_matches(matches, query_point_cloud)
            
            if not pose_result:
                return jsonify({'error': 'Pose estimation failed'}), 500
            
            # Create overlay
            overlay_path = self.create_pose_overlay(pose_result)
            
            return jsonify({
                'pose_estimation': pose_result,
                'overlay_path': overlay_path,
                'num_matches': len(matches),
                'processing_time': time.time()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def handle_target_features_loading(self):
        """Handle target features loading request"""
        try:
            if 'target_features_zip' not in request.files:
                return jsonify({'error': 'Target features ZIP file required'}), 400
            
            target_zip_file = request.files['target_features_zip']
            
            # Save temporarily
            temp_path = f"/tmp/target_features_{int(time.time())}.zip"
            target_zip_file.save(temp_path)
            
            # Load features
            success = self.load_target_features_from_zip(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            if success:
                return jsonify({
                    'message': 'Target features loaded successfully',
                    'num_features': len(self.target_features) if self.target_features is not None else 0
                })
            else:
                return jsonify({'error': 'Failed to load target features'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def run_flask_server(self):
        """Run Flask server"""
        self.app.run(host='0.0.0.0', port=8002, debug=False)
    
    def run(self):
        """Run the pose overlay service"""
        print("🚀 Starting Pose Overlay Pipeline Service")
        print(f"💻 CPU Cores: {self.system_resources['cpu_count']}")
        print(f"📊 Memory: {self.system_resources['memory_total']:.1f}GB")
        print("🌐 Starting Flask server on port 8002...")
        
        try:
            self.run_flask_server()
        except KeyboardInterrupt:
            print("🛑 Shutting down Pose Overlay Pipeline")
        finally:
            self.executor.shutdown(wait=True)

if __name__ == "__main__":
    pipeline = PoseOverlayPipeline()
    pipeline.run()
