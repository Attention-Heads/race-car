#!/usr/bin/env python3
"""
Create Velocity Scaler Script

This script creates and saves a velocity scaler based on the expert training data.
Run this script before training your model to ensure consistent velocity preprocessing.
"""

import pandas as pd
import argparse
import sys
from preprocessing_utils import StatePreprocessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_scaler(data_path: str = "processed_balanced_training_data.csv", 
                 scaler_path: str = "velocity_scaler.pkl",
                 use_filtered_data: bool = True):
    """
    Create and save a velocity scaler from training data.
    
    Args:
        data_path: Path to the expert training data CSV
        scaler_path: Path where to save the velocity scaler
    """
    
    print("=" * 60)
    print("CREATING VELOCITY SCALER")
    print("=" * 60)
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} rows from {data_path}")
    except Exception as e:
        print(f"✗ Failed to load data from {data_path}: {e}")
        return False
    
    # Check if we have velocity data
    required_cols = ['velocity_x', 'velocity_y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return False
    
    # Show velocity statistics
    print(f"\nVelocity Statistics:")
    print(f"  velocity_x: min={df['velocity_x'].min():.3f}, max={df['velocity_x'].max():.3f}, mean={df['velocity_x'].mean():.3f}, std={df['velocity_x'].std():.3f}")
    print(f"  velocity_y: min={df['velocity_y'].min():.3f}, max={df['velocity_y'].max():.3f}, mean={df['velocity_y'].mean():.3f}, std={df['velocity_y'].std():.3f}")
    
    # Create the preprocessor and scaler
    print(f"\nCreating velocity scaler...")
    preprocessor = StatePreprocessor(use_velocity_scaler=False, velocity_scaler_path=scaler_path)
    
    try:
        preprocessor.create_velocity_scaler(df)
        print(f"✓ Velocity scaler created and saved to {scaler_path}")
        
        # Verify the scaler was created correctly
        if preprocessor.velocity_scaler is not None:
            print(f"✓ Scaler verification:")
            print(f"  Mean: {preprocessor.velocity_scaler.mean_}")
            print(f"  Scale (std): {preprocessor.velocity_scaler.scale_}")
        else:
            print(f"⚠ Warning: Scaler was not properly created")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create velocity scaler: {e}")
        return False
    
    # Test the scaler by preprocessing a few samples
    print(f"\nTesting scaler with sample data...")
    try:
        sample_rows = df.sample(n=min(5, len(df)), random_state=42)
        
        print(f"Sample preprocessing results:")
        for i, (_, row) in enumerate(sample_rows.iterrows()):
            # Create a new preprocessor that loads the saved scaler
            test_preprocessor = StatePreprocessor(use_velocity_scaler=True, velocity_scaler_path=scaler_path)
            processed_state = test_preprocessor.preprocess_dataframe_row(row)
            
            print(f"  Row {i+1}: Raw velocity=({row['velocity_x']:.3f}, {row['velocity_y']:.3f}) -> Scaled=({processed_state[0]:.3f}, {processed_state[1]:.3f})")
        
        print(f"✓ Scaler test completed successfully")
        
    except Exception as e:
        print(f"⚠ Scaler test failed: {e}")
        return False
    
    print(f"\n" + "=" * 60)
    print(f"SUCCESS: Velocity scaler created at {scaler_path}")
    print(f"You can now run training and inference with consistent velocity preprocessing.")
    print(f"=" * 60)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create velocity scaler from expert data")
    parser.add_argument("--data", default="processed_balanced_training_data.csv",
                       help="Path to expert training data CSV (default: processed_balanced_training_data.csv)")
    parser.add_argument("--output", default="velocity_scaler.pkl", 
                       help="Output path for velocity scaler (default: velocity_scaler.pkl)")
    parser.add_argument("--include-nothing", action="store_true",
                       help="Include NOTHING actions in scaler calculation (default: filter them out)")
    
    args = parser.parse_args()
    
    success = create_scaler(
        data_path=args.data,
        scaler_path=args.output,
        use_filtered_data=not args.include_nothing
    )
    
    if not success:
        print(f"\nFailed to create velocity scaler. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
