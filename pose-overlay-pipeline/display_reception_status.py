#!/usr/bin/env python3

import json
import time
from pathlib import Path
from feature_vector_receiver import FeatureVectorReceiver

def display_current_status():
    """Display current feature reception status"""
    receiver = FeatureVectorReceiver()
    
    print("=" * 60)
    print("🎯 POSE OVERLAY PIPELINE - FEATURE RECEPTION STATUS")
    print("=" * 60)
    
    # Generate and display status
    status_message = receiver.generate_status_message()
    print(status_message)
    
    # Save status to file
    status_file = receiver.save_reception_status()
    print(f"\n📝 Detailed status saved to: {status_file}")
    
    # Display validation results
    validation = receiver.validate_features()
    print(f"\n🔍 VALIDATION RESULTS:")
    print(f"   Target Received: {'✅' if validation['target_received'] else '❌'}")
    print(f"   Query Received: {'✅' if validation['query_received'] else '❌'}")
    print(f"   Compatible: {'✅' if validation['compatible'] else '❌'}")
    print(f"   Ready for Pose Estimation: {'✅' if validation['ready_for_pose_estimation'] else '❌'}")
    
    print("=" * 60)
    
    return validation

if __name__ == "__main__":
    display_current_status()
