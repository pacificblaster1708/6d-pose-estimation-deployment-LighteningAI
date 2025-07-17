import os
import json
import time
import zipfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

class FeatureVectorReceiver:
    """Handler for receiving and processing feature vectors from target and query pipelines"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        
        # Storage for received features
        self.target_features = None  # Will be 1000×128
        self.query_features = None   # Will be 5000×128
        
        # Metadata
        self.target_metadata = {}
        self.query_metadata = {}
        
        self.logger.info("🎯 Feature Vector Receiver initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/app/logs/feature_receiver.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            '/app/data/target_features',
            '/app/data/query_features',
            '/app/output/pose_results',
            '/app/logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def receive_target_features(self, target_zip_path: str) -> bool:
        """
        Receive and process target pipeline features
        Expected: ZIP file containing 1000×128 dimensional vectors
        """
        try:
            self.logger.info(f"📦 Receiving target features from: {target_zip_path}")
            
            if not os.path.exists(target_zip_path):
                raise FileNotFoundError(f"Target ZIP file not found: {target_zip_path}")
            
            # Extract and process target features
            with zipfile.ZipFile(target_zip_path, 'r') as zip_ref:
                extract_path = Path('/app/data/target_features/extracted')
                extract_path.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(extract_path)
                
                # Find all .npy files
                feature_files = list(extract_path.glob('*.npy'))
                
                if not feature_files:
                    raise ValueError("No .npy files found in target ZIP")
                
                # Load and concatenate features
                features_list = []
                metadata_list = []
                
                for feature_file in feature_files:
                    features = np.load(feature_file)
                    
                    # Validate dimensions
                    if features.shape[1] != 128:
                        raise ValueError(f"Invalid feature dimensions: {features.shape}, expected Nx128")
                    
                    features_list.append(features)
                    metadata_list.append({
                        'filename': feature_file.name,
                        'shape': features.shape,
                        'frame_count': features.shape[0]
                    })
                
                # Concatenate all features
                self.target_features = np.vstack(features_list)
                
                # Validate final dimensions
                if self.target_features.shape[1] != 128:
                    raise ValueError(f"Target features must have 128 dimensions, got {self.target_features.shape[1]}")
                
                # Store metadata
                self.target_metadata = {
                    'total_vectors': self.target_features.shape[0],
                    'expected_vectors': 1000,
                    'dimensions': self.target_features.shape[1],
                    'files_processed': len(feature_files),
                    'file_details': metadata_list,
                    'received_timestamp': time.time(),
                    'status': 'received'
                }
                
                self.logger.info(f"✅ Target features received successfully!")
                self.logger.info(f"   📊 Shape: {self.target_features.shape}")
                self.logger.info(f"   🎯 Expected: (~1000, 128)")
                self.logger.info(f"   📁 Files processed: {len(feature_files)}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to receive target features: {e}")
            return False
    
    def receive_query_features(self, query_features_path: str) -> bool:
        """
        Receive and process query pipeline features
        Expected: Single .npy file with 5000×128 dimensional vectors
        """
        try:
            self.logger.info(f"📦 Receiving query features from: {query_features_path}")
            
            if not os.path.exists(query_features_path):
                raise FileNotFoundError(f"Query features file not found: {query_features_path}")
            
            # Load query features
            self.query_features = np.load(query_features_path)
            
            # Validate dimensions
            if len(self.query_features.shape) != 2:
                raise ValueError(f"Query features must be 2D array, got shape: {self.query_features.shape}")
            
            if self.query_features.shape[1] != 128:
                raise ValueError(f"Query features must have 128 dimensions, got {self.query_features.shape[1]}")
            
            # Store metadata
            self.query_metadata = {
                'total_vectors': self.query_features.shape[0],
                'expected_vectors': 5000,
                'dimensions': self.query_features.shape[1],
                'received_timestamp': time.time(),
                'status': 'received'
            }
            
            self.logger.info(f"✅ Query features received successfully!")
            self.logger.info(f"   📊 Shape: {self.query_features.shape}")
            self.logger.info(f"   🎯 Expected: (~5000, 128)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to receive query features: {e}")
            return False
    
    def validate_features(self) -> Dict[str, Any]:
        """Validate that both feature sets have been received and are compatible"""
        validation_result = {
            'target_received': self.target_features is not None,
            'query_received': self.query_features is not None,
            'compatible': False,
            'ready_for_pose_estimation': False,
            'details': {}
        }
        
        if validation_result['target_received']:
            validation_result['details']['target'] = {
                'shape': self.target_features.shape,
                'metadata': self.target_metadata
            }
        
        if validation_result['query_received']:
            validation_result['details']['query'] = {
                'shape': self.query_features.shape,
                'metadata': self.query_metadata
            }
        
        # Check compatibility
        if validation_result['target_received'] and validation_result['query_received']:
            target_dims = self.target_features.shape[1]
            query_dims = self.query_features.shape[1]
            
            validation_result['compatible'] = (target_dims == query_dims == 128)
            validation_result['ready_for_pose_estimation'] = validation_result['compatible']
            
            if validation_result['compatible']:
                self.logger.info("✅ Both feature sets received and compatible!")
                self.logger.info(f"   🎯 Target: {self.target_features.shape}")
                self.logger.info(f"   🔍 Query: {self.query_features.shape}")
                self.logger.info("   🚀 Ready for pose estimation processing")
            else:
                self.logger.error("❌ Feature sets are not compatible!")
                self.logger.error(f"   Target dimensions: {target_dims}")
                self.logger.error(f"   Query dimensions: {query_dims}")
        
        return validation_result
    
    def save_reception_status(self) -> str:
        """Save the current reception status to a file"""
        status_data = {
            'timestamp': time.time(),
            'target_features': {
                'received': self.target_features is not None,
                'shape': self.target_features.shape if self.target_features is not None else None,
                'metadata': self.target_metadata
            },
            'query_features': {
                'received': self.query_features is not None,
                'shape': self.query_features.shape if self.query_features is not None else None,
                'metadata': self.query_metadata
            },
            'validation': self.validate_features(),
            'message': self.generate_status_message()
        }
        
        # Save to file
        status_file = Path('/app/output/feature_reception_status.json')
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
        
        self.logger.info(f"📝 Status saved to: {status_file}")
        return str(status_file)
    
    def generate_status_message(self) -> str:
        """Generate a human-readable status message"""
        if self.target_features is not None and self.query_features is not None:
            return (
                f"✅ FEATURE RECEPTION COMPLETE\n"
                f"📦 Target Features: {self.target_features.shape[0]}×{self.target_features.shape[1]} vectors received\n"
                f"🔍 Query Features: {self.query_features.shape[0]}×{self.query_features.shape[1]} vectors received\n"
                f"🚀 Ready for 6D pose estimation processing"
            )
        elif self.target_features is not None:
            return (
                f"⏳ PARTIAL RECEPTION\n"
                f"✅ Target Features: {self.target_features.shape[0]}×{self.target_features.shape[1]} vectors received\n"
                f"⏳ Query Features: Waiting for input\n"
                f"🔄 Pose estimation pending query features"
            )
        elif self.query_features is not None:
            return (
                f"⏳ PARTIAL RECEPTION\n"
                f"⏳ Target Features: Waiting for input\n"
                f"✅ Query Features: {self.query_features.shape[0]}×{self.query_features.shape[1]} vectors received\n"
                f"🔄 Pose estimation pending target features"
            )
        else:
            return (
                f"⏳ WAITING FOR INPUTS\n"
                f"📦 Target Features: Not received (expecting ~1000×128)\n"
                f"🔍 Query Features: Not received (expecting ~5000×128)\n"
                f"🔄 Ready to receive feature vectors"
            )
    
    def process_received_features(self) -> Dict[str, Any]:
        """Process the received features for pose estimation"""
        validation = self.validate_features()
        
        if not validation['ready_for_pose_estimation']:
            return {
                'success': False,
                'error': 'Features not ready for processing',
                'validation': validation
            }
        
        try:
            # Save processed features for pose estimation
            processed_dir = Path('/app/output/processed_features')
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save target features
            target_processed_path = processed_dir / 'target_features_processed.npy'
            np.save(target_processed_path, self.target_features)
            
            # Save query features
            query_processed_path = processed_dir / 'query_features_processed.npy'
            np.save(query_processed_path, self.query_features)
            
            # Create processing metadata
            processing_metadata = {
                'target_features_path': str(target_processed_path),
                'query_features_path': str(query_processed_path),
                'target_shape': self.target_features.shape,
                'query_shape': self.query_features.shape,
                'processing_timestamp': time.time(),
                'ready_for_pose_estimation': True
            }
            
            # Save metadata
            metadata_path = processed_dir / 'processing_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(processing_metadata, f, indent=2, default=str)
            
            self.logger.info("✅ Features processed and ready for pose estimation")
            self.logger.info(f"   📁 Target features: {target_processed_path}")
            self.logger.info(f"   📁 Query features: {query_processed_path}")
            
            return {
                'success': True,
                'metadata': processing_metadata,
                'validation': validation
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process features: {e}")
            return {
                'success': False,
                'error': str(e),
                'validation': validation
            }

# Example usage
if __name__ == "__main__":
    # Initialize receiver
    receiver = FeatureVectorReceiver()
    
    # Example of receiving features
    print("🎯 Feature Vector Receiver - Ready to receive inputs")
    print("📦 Waiting for target features ZIP (1000×128 vectors)")
    print("🔍 Waiting for query features file (5000×128 vectors)")
    
    # Save initial status
    receiver.save_reception_status()
    
    # Display current status
    print("\n" + receiver.generate_status_message())
