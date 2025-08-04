"""
Data Analysis Script for Race Car Imitation Learning

This script helps analyze the logged game data for training imitation learning models.
"""

import json
import os
import pandas as pd
import numpy as np
import shutil
from typing import List, Dict, Any, Callable

def load_game_data(data_dir: str = "data") -> List[Dict[str, Any]]:
    """
    Load all game data files from the data directory.
    
    :param data_dir: Directory containing the game data files
    :return: List of game data dictionaries
    """
    games = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return games
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and filename.startswith('game_'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                    games.append(game_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return games

def create_training_dataset(games: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert game data into a pandas DataFrame suitable for machine learning.
    
    :param games: List of game data dictionaries
    :return: DataFrame with features and actions
    """
    training_data = []
    
    for game in games:
        game_steps = game.get('game_data', [])
        
        for step in game_steps:
            env_state = step.get('environment_state', {})
            action = step.get('action', 'NOTHING')
            
            # Extract features
            row = {
                'game_number': game.get('game_number', 0),
                'tick': env_state.get('tick', 0),
                'distance': env_state.get('distance', 0),
                'crashed': env_state.get('crashed', False),
                'action': action
            }
            
            # Add ego car state
            ego_car = env_state.get('ego_car', {})
            ego_pos = ego_car.get('position', {})
            ego_vel = ego_car.get('velocity', {})
            
            row.update({
                'ego_x': ego_pos.get('x', 0),
                'ego_y': ego_pos.get('y', 0),
                'ego_vel_x': ego_vel.get('x', 0),
                'ego_vel_y': ego_vel.get('y', 0)
            })
            
            # Add sensor readings
            sensors = env_state.get('sensors', {})
            for sensor_name, reading in sensors.items():
                row[f'sensor_{sensor_name}'] = reading if reading is not None else 1000.0
            
            # Add information about other cars (simplified)
            other_cars = env_state.get('other_cars', [])
            row['num_other_cars'] = len(other_cars)
            
            # Find closest car in each direction (simplified)
            closest_car_distance = 1000.0
            for car in other_cars:
                car_pos = car.get('position', {})
                car_x = car_pos.get('x', 0)
                distance = abs(car_x - ego_pos.get('x', 0))
                if distance < closest_car_distance:
                    closest_car_distance = distance
            
            row['closest_car_distance'] = closest_car_distance
            
            training_data.append(row)
    
    return pd.DataFrame(training_data)

def analyze_games(games: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics about the games.
    
    :param games: List of game data dictionaries
    """
    if not games:
        print("No games to analyze.")
        return
    
    print(f"\n=== Game Analysis ===")
    print(f"Total games loaded: {len(games)}")
    
    # Game statistics
    scores = [game.get('final_score', 0) for game in games]
    ticks = [game.get('final_tick', 0) for game in games]
    crashed_games = sum(1 for game in games if game.get('crashed', False))
    
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Best score: {max(scores):.2f}")
    print(f"Worst score: {min(scores):.2f}")
    print(f"Average game duration (ticks): {np.mean(ticks):.1f}")
    print(f"Crashed games: {crashed_games}/{len(games)} ({100*crashed_games/len(games):.1f}%)")
    
    # Action distribution
    all_actions = []
    for game in games:
        game_steps = game.get('game_data', [])
        for step in game_steps:
            action = step.get('action', 'NOTHING')
            all_actions.append(action)
    
    if all_actions:
        action_counts = pd.Series(all_actions).value_counts()
        print(f"\nAction distribution:")
        for action, count in action_counts.items():
            percentage = 100 * count / len(all_actions)
            print(f"  {action}: {count} ({percentage:.1f}%)")

def export_for_ml(games: List[Dict[str, Any]], output_file: str = "training_data.csv") -> None:
    """
    Export game data to a CSV file for machine learning.
    
    :param games: List of game data dictionaries
    :param output_file: Output CSV filename
    """
    df = create_training_dataset(games)
    
    if not df.empty:
        df.to_csv(output_file, index=False)
        print(f"Training data exported to {output_file}")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
    else:
        print("No data to export.")

def copy_runs_by_criteria(
    source_dir: str = "data",
    target_dir: str = "filtered_data",
    min_distance: float = None,
    max_distance: float = None,
    min_score: float = None,
    max_score: float = None,
    crashed_only: bool = None,
    not_crashed_only: bool = None,
    min_steps: int = None,
    max_steps: int = None,
    custom_filter: Callable[[Dict[str, Any]], bool] = None
) -> int:
    """
    Copy race car runs that meet specified criteria to a new directory.
    
    This function filters game runs based on various criteria and copies both
    the CSV data files and JSON metadata files to a target directory. If files
    already exist in the target directory, they will be overwritten.
    
    :param source_dir: Source directory containing the game data files
    :param target_dir: Target directory where filtered files will be copied
    :param min_distance: Minimum distance traveled (from final step in CSV)
    :param max_distance: Maximum distance traveled
    :param min_score: Minimum final score
    :param max_score: Maximum final score
    :param crashed_only: If True, only copy runs that crashed
    :param not_crashed_only: If True, only copy runs that didn't crash
    :param min_steps: Minimum number of steps/ticks
    :param max_steps: Maximum number of steps/ticks
    :param custom_filter: Custom function that takes metadata dict and returns bool
    :return: Number of runs copied
    """
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return 0
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    copied_count = 0
    processed_count = 0
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(source_dir) 
                     if f.endswith('_metadata.json') and f.startswith('game_')]
    
    print(f"Found {len(metadata_files)} metadata files to process...")
    
    for metadata_file in metadata_files:
        processed_count += 1
        metadata_path = os.path.join(source_dir, metadata_file)
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get corresponding CSV file
            csv_file = metadata_file.replace('_metadata.json', '.csv')
            csv_path = os.path.join(source_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file {csv_file} not found for {metadata_file}")
                continue
            
            # Get distance from CSV file (last row's distance value)
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and 'distance' in df.columns:
                    final_distance = df['distance'].iloc[-1]
                else:
                    final_distance = 0
            except Exception as e:
                print(f"Warning: Could not read distance from {csv_file}: {e}")
                final_distance = 0
            
            # Apply filters
            meets_criteria = True
            
            # Distance filters
            if min_distance is not None and final_distance < min_distance:
                meets_criteria = False
            if max_distance is not None and final_distance > max_distance:
                meets_criteria = False
            
            # Score filters
            if min_score is not None and metadata.get('final_score', 0) < min_score:
                meets_criteria = False
            if max_score is not None and metadata.get('final_score', 0) > max_score:
                meets_criteria = False
            
            # Crash filters
            if crashed_only is True and not metadata.get('crashed', False):
                meets_criteria = False
            if not_crashed_only is True and metadata.get('crashed', False):
                meets_criteria = False
            
            # Steps filters
            total_steps = metadata.get('total_steps', 0)
            if min_steps is not None and total_steps < min_steps:
                meets_criteria = False
            if max_steps is not None and total_steps > max_steps:
                meets_criteria = False
            
            # Custom filter
            if custom_filter is not None and not custom_filter(metadata):
                meets_criteria = False
            
            # Copy files if criteria are met
            if meets_criteria:
                # Copy metadata file
                target_metadata_path = os.path.join(target_dir, metadata_file)
                shutil.copy2(metadata_path, target_metadata_path)
                
                # Copy CSV file
                target_csv_path = os.path.join(target_dir, csv_file)
                shutil.copy2(csv_path, target_csv_path)
                
                copied_count += 1
                print(f"Copied run {copied_count}: {csv_file} (distance: {final_distance:.1f}, score: {metadata.get('final_score', 0):.1f}, crashed: {metadata.get('crashed', False)})")
        
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
    
    print(f"\nFiltering complete!")
    print(f"Processed: {processed_count} runs")
    print(f"Copied: {copied_count} runs to '{target_dir}'")
    
    return copied_count

def copy_good_runs_for_imitation_learning(
    source_dir: str = "data",
    target_dir: str = "good_runs",
    min_distance: float = 20000,
    not_crashed_only: bool = True
) -> int:
    """
    Convenience function to copy good runs for imitation learning.
    
    This filters for runs with good distance and no crashes, which are
    typically the best candidates for imitation learning.
    
    :param source_dir: Source directory containing the game data files
    :param target_dir: Target directory where good runs will be copied
    :param min_distance: Minimum distance to consider a "good" run
    :param not_crashed_only: Only copy runs that didn't crash
    :return: Number of runs copied
    """
    return copy_runs_by_criteria(
        source_dir=source_dir,
        target_dir=target_dir,
        min_distance=min_distance,
        not_crashed_only=not_crashed_only
    )

def analyze_run_quality(source_dir: str = "data") -> Dict[str, Any]:
    """
    Analyze the quality distribution of runs to help determine good filtering criteria.
    
    :param source_dir: Directory containing the game data files
    :return: Dictionary with analysis results
    """
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' not found.")
        return {}
    
    runs_data = []
    
    # Get all metadata files
    metadata_files = [f for f in os.listdir(source_dir) 
                     if f.endswith('_metadata.json') and f.startswith('game_')]
    
    for metadata_file in metadata_files:
        metadata_path = os.path.join(source_dir, metadata_file)
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get corresponding CSV file
            csv_file = metadata_file.replace('_metadata.json', '.csv')
            csv_path = os.path.join(source_dir, csv_file)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and 'distance' in df.columns:
                        final_distance = df['distance'].iloc[-1]
                    else:
                        final_distance = 0
                except:
                    final_distance = 0
            else:
                final_distance = 0
            
            runs_data.append({
                'file': metadata_file,
                'distance': final_distance,
                'score': metadata.get('final_score', 0),
                'crashed': metadata.get('crashed', False),
                'steps': metadata.get('total_steps', 0),
                'elapsed_time': metadata.get('elapsed_time', 0)
            })
        
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
    
    if not runs_data:
        return {}
    
    df = pd.DataFrame(runs_data)
    
    analysis = {
        'total_runs': len(runs_data),
        'crashed_runs': df['crashed'].sum(),
        'successful_runs': (~df['crashed']).sum(),
        'distance_stats': {
            'mean': df['distance'].mean(),
            'median': df['distance'].median(),
            'min': df['distance'].min(),
            'max': df['distance'].max(),
            'std': df['distance'].std(),
            'percentiles': {
                '25%': df['distance'].quantile(0.25),
                '75%': df['distance'].quantile(0.75),
                '90%': df['distance'].quantile(0.90),
                '95%': df['distance'].quantile(0.95)
            }
        },
        'score_stats': {
            'mean': df['score'].mean(),
            'median': df['score'].median(),
            'min': df['score'].min(),
            'max': df['score'].max(),
            'std': df['score'].std()
        }
    }
    
    # Print analysis
    print("\nRun Quality Analysis")
    print("===================")
    print(f"Total runs: {analysis['total_runs']}")
    print(f"Crashed runs: {analysis['crashed_runs']} ({analysis['crashed_runs']/analysis['total_runs']*100:.1f}%)")
    print(f"Successful runs: {analysis['successful_runs']} ({analysis['successful_runs']/analysis['total_runs']*100:.1f}%)")
    print("\nDistance Statistics:")
    print(f"  Mean: {analysis['distance_stats']['mean']:.1f}")
    print(f"  Median: {analysis['distance_stats']['median']:.1f}")
    print(f"  Min: {analysis['distance_stats']['min']:.1f}")
    print(f"  Max: {analysis['distance_stats']['max']:.1f}")
    print(f"  75th percentile: {analysis['distance_stats']['percentiles']['75%']:.1f}")
    print(f"  90th percentile: {analysis['distance_stats']['percentiles']['90%']:.1f}")
    print(f"  95th percentile: {analysis['distance_stats']['percentiles']['95%']:.1f}")
    
    return analysis

def main():
    """
    Main function to analyze game data.
    """
    print("Race Car Data Analysis")
    print("=====================")
    
    # Load game data
    games = load_game_data()
    
    if not games:
        print("No game data found. Make sure to play some games first!")
        return
    
    # Analyze games
    analyze_games(games)
    
    # Analyze run quality to help determine filtering criteria
    analysis = analyze_run_quality()
    
    # Export for ML
    export_for_ml(games)
    
    print("\nAnalysis complete!")
    
    # Example usage of filtering functions
    print("\nExample filtering operations:")
    print("1. To copy good runs for imitation learning:")
    print("   copy_good_runs_for_imitation_learning(min_distance=20000)")
    print("2. To copy runs with custom criteria:")
    print("   copy_runs_by_criteria(target_dir='custom_filter', min_distance=15000, not_crashed_only=True)")

if __name__ == "__main__":
    main()
