"""
Expert Data Processing Script for Race Car Imitation Learning

This script implements the critical foundation for building a race car AI:
1. Filters out bad behavior using N-Step Horizon Cutoff
2. Normalizes sensor and velocity data properly
3. Creates clean expert dataset for training

Based on the strategy that crashed games contain bad decisions leading up to the crash,
this script removes the last N timesteps before any crash to create expert data.
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Any, Tuple

class ExpertDataProcessor:
    def __init__(self, data_dir: str = "data", n_steps_to_remove: int = 150):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing CSV and JSON game data files
            n_steps_to_remove: Number of timesteps to remove before crashes (hyperparameter)
        """
        self.data_dir = data_dir
        self.n_steps_to_remove = n_steps_to_remove
        self.velocity_scaler = StandardScaler()
        
        # Define sensor columns based on README
        self.sensor_cols = [
            'sensor_left_side', 'sensor_left_side_front', 'sensor_left_front',
            'sensor_front_left_front', 'sensor_front', 'sensor_front_right_front',
            'sensor_right_front', 'sensor_right_side_front', 'sensor_right_side',
            'sensor_right_side_back', 'sensor_right_back', 'sensor_back_right_back',
            'sensor_back', 'sensor_back_left_back', 'sensor_left_back', 'sensor_left_side_back'
        ]
        
        self.velocity_cols = ['velocity_x', 'velocity_y']
        
    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load all metadata files to understand which games crashed."""
        metadata_files = glob.glob(os.path.join(self.data_dir, "*_metadata.json"))
        metadata_list = []
        
        for filepath in metadata_files:
            try:
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    # Extract game identifier from filename
                    filename = os.path.basename(filepath)
                    game_id = filename.replace('_metadata.json', '')
                    metadata['game_id'] = game_id
                    metadata_list.append(metadata)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return metadata_list
    
    def load_all_game_data(self) -> pd.DataFrame:
        """Load and combine all CSV game data files."""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        dataframes = []
        
        for filepath in csv_files:
            try:
                # Extract game identifier from filename
                filename = os.path.basename(filepath)
                game_id = filename.replace('.csv', '')
                
                df = pd.read_csv(filepath)
                df['game_id'] = game_id
                dataframes.append(df)
                print(f"Loaded {len(df)} rows from {filename}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        if not dataframes:
            raise ValueError("No CSV files found in data directory!")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} total rows from {len(dataframes)} games")
        return combined_df
    
    def analyze_crash_patterns(self, df: pd.DataFrame, metadata: List[Dict]) -> None:
        """Analyze crash patterns in the data."""
        crashed_games = [m for m in metadata if m['crashed']]
        successful_games = [m for m in metadata if not m['crashed']]
        
        print(f"\n=== CRASH ANALYSIS ===")
        print(f"Total games: {len(metadata)}")
        print(f"Crashed games: {len(crashed_games)}")
        print(f"Successful games: {len(successful_games)}")
        
        if successful_games:
            print(f"\nSuccessful games (GOLDEN DATA):")
            for game in successful_games:
                print(f"  {game['game_id']}: Score={game['final_score']:.0f}, Ticks={game['final_tick']}")
        
        if crashed_games:
            avg_crash_tick = np.mean([g['final_tick'] for g in crashed_games])
            print(f"\nCrashed games average tick: {avg_crash_tick:.0f}")
            print(f"Will remove last {self.n_steps_to_remove} steps from crashed games")
    
    def filter_bad_behavior(self, df: pd.DataFrame, metadata: List[Dict]) -> pd.DataFrame:
        """
        Apply N-Step Horizon Cutoff to remove bad decisions leading to crashes.
        
        This is the CRITICAL step - removes the sequence of bad decisions that led to crashes.
        """
        print(f"\n=== APPLYING N-STEP HORIZON CUTOFF (N={self.n_steps_to_remove}) ===")
        
        # Create mapping of game_id to crash status
        crash_info = {m['game_id']: m['crashed'] for m in metadata}
        
        indices_to_drop = []
        original_size = len(df)
        
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id].copy()
            
            if crash_info.get(game_id, False):  # Game crashed
                # Find when crash occurred (did_crash becomes True)
                crash_rows = game_df[game_df['did_crash'] == True]
                
                if len(crash_rows) > 0:
                    # Get the first tick where crash occurred
                    crash_tick = crash_rows['elapsed_ticks'].min()
                    
                    # Remove last N steps before crash + the crash itself
                    cutoff_tick = max(1, crash_tick - self.n_steps_to_remove)
                    
                    # Mark rows for removal
                    rows_to_remove = game_df[game_df['elapsed_ticks'] >= cutoff_tick].index
                    indices_to_drop.extend(rows_to_remove)
                    
                    kept_rows = len(game_df) - len(rows_to_remove)
                    print(f"  {game_id}: Crashed at tick {crash_tick}, keeping {kept_rows}/{len(game_df)} rows")
                else:
                    print(f"  {game_id}: Marked as crashed but no crash detected in data")
            else:  # Successful game - keep all data
                print(f"  {game_id}: Successful run, keeping all {len(game_df)} rows (GOLDEN DATA)")
        
        # Apply the filter
        expert_df = df.drop(indices_to_drop).copy()
        
        print(f"\nFiltering results:")
        print(f"  Original data points: {original_size:,}")
        print(f"  Removed data points: {len(indices_to_drop):,}")
        print(f"  Expert data points: {len(expert_df):,}")
        print(f"  Retention rate: {len(expert_df)/original_size*100:.1f}%")
        
        return expert_df
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sensor and velocity features for neural network training.
        
        Sensors: Scale to [0,1] by dividing by 1000 (known max range)
        Velocity: Z-score normalization (standardization)
        """
        print(f"\n=== FEATURE NORMALIZATION ===")
        
        expert_df = df.copy()
        
        # 1. Handle sensor data
        print("Processing sensor data...")
        
        # Check which sensor columns actually exist in the data
        existing_sensor_cols = [col for col in self.sensor_cols if col in expert_df.columns]
        print(f"  Found {len(existing_sensor_cols)}/{len(self.sensor_cols)} sensor columns")
        
        if existing_sensor_cols:
            # Fill NaN values with 1000 (indicating open space/no obstacle detected)
            nan_counts_before = expert_df[existing_sensor_cols].isnull().sum().sum()
            expert_df[existing_sensor_cols] = expert_df[existing_sensor_cols].fillna(1000.0)
            print(f"  Filled {nan_counts_before} NaN sensor values with 1000.0")
            
            # Scale sensors to [0, 1] range
            expert_df[existing_sensor_cols] = expert_df[existing_sensor_cols] / 1000.0
            print(f"  Scaled sensor values to [0,1] range")
            
            # Verify scaling
            sensor_min = expert_df[existing_sensor_cols].min().min()
            sensor_max = expert_df[existing_sensor_cols].max().max()
            print(f"  Sensor range after scaling: [{sensor_min:.3f}, {sensor_max:.3f}]")
        
        # 2. Handle velocity data
        print("\nProcessing velocity data...")
        
        existing_velocity_cols = [col for col in self.velocity_cols if col in expert_df.columns]
        print(f"  Found velocity columns: {existing_velocity_cols}")
        
        if existing_velocity_cols:
            # Calculate statistics before normalization
            velocity_stats_before = expert_df[existing_velocity_cols].describe()
            print("  Velocity statistics before normalization:")
            print(velocity_stats_before)
            
            # Apply standardization (Z-score normalization)
            expert_df[existing_velocity_cols] = self.velocity_scaler.fit_transform(
                expert_df[existing_velocity_cols]
            )
            
            # Save the scaler for later use during inference
            scaler_path = 'velocity_scaler.pkl'
            joblib.dump(self.velocity_scaler, scaler_path)
            print(f"  Saved velocity scaler to: {scaler_path}")
            
            # Verify normalization
            velocity_stats_after = expert_df[existing_velocity_cols].describe()
            print("  Velocity statistics after normalization:")
            print(velocity_stats_after)
        
        return expert_df
    
    def create_expert_dataset(self, save_path: str = 'expert_training_data.csv') -> pd.DataFrame:
        """
        Main method to create the expert dataset with all processing steps.
        """
        print("=" * 60)
        print("CREATING EXPERT DATASET FOR RACE CAR AI")
        print("=" * 60)
        
        # Step 1: Load metadata to understand crash patterns
        metadata = self.load_metadata()
        
        # Step 2: Load all game data
        df = self.load_all_game_data()
        
        # Step 3: Analyze crash patterns
        self.analyze_crash_patterns(df, metadata)
        
        # Step 4: Filter bad behavior (THE CRITICAL STEP)
        expert_df = self.filter_bad_behavior(df, metadata)
        
        # Step 5: Normalize features
        expert_df = self.normalize_features(expert_df)
        
        # Step 6: Save the expert dataset
        expert_df.to_csv(save_path, index=False)
        print(f"\n=== EXPERT DATASET SAVED ===")
        print(f"Saved to: {save_path}")
        print(f"Shape: {expert_df.shape}")
        print(f"Columns: {list(expert_df.columns)}")
        
        # Step 7: Final summary
        self.print_final_summary(expert_df)
        
        return expert_df
    
    def print_final_summary(self, expert_df: pd.DataFrame) -> None:
        """Print final summary of the expert dataset."""
        print(f"\n=== EXPERT DATASET SUMMARY ===")
        print(f"Total training examples: {len(expert_df):,}")
        print(f"Unique games: {expert_df['game_id'].nunique()}")
        print(f"Action distribution:")
        
        action_counts = expert_df['action'].value_counts()
        for action, count in action_counts.items():
            percentage = count / len(expert_df) * 100
            print(f"  {action}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nFeature ranges:")
        # Check sensor ranges
        sensor_cols_present = [col for col in expert_df.columns if 'sensor_' in col]
        if sensor_cols_present:
            sensor_min = expert_df[sensor_cols_present].min().min()
            sensor_max = expert_df[sensor_cols_present].max().max()
            print(f"  Sensors: [{sensor_min:.3f}, {sensor_max:.3f}]")
        
        # Check velocity ranges
        velocity_cols_present = [col for col in expert_df.columns if 'velocity_' in col]
        if velocity_cols_present:
            velocity_stats = expert_df[velocity_cols_present].describe()
            print(f"  Velocities (normalized): mean≈0, std≈1")
        
        print(f"\nDataset ready for neural network training! 🚗💨")

def main():
    """Example usage of the ExpertDataProcessor."""
    
    # Initialize processor with hyperparameters
    processor = ExpertDataProcessor(
        data_dir="data",
        n_steps_to_remove=150  # Experiment with 100, 150, 200
    )
    
    # Create expert dataset
    expert_df = processor.create_expert_dataset('expert_training_data.csv')
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Use 'expert_training_data.csv' for training your neural network")
    print("2. Load 'velocity_scaler.pkl' during inference for velocity normalization")
    print("3. Experiment with different n_steps_to_remove values (100, 150, 200)")
    print("4. Consider log transform on sensors if model is insensitive to nearby objects")
    print("="*60)

if __name__ == "__main__":
    main()
