"""
Data Processing Script for Race Car Imitation Learning

This script processes raw game data from CSV files to prepare a clean,
balanced dataset for training an imitation learning model.

It performs the following steps:
1. Iterates through all game run files in the specified data directory.
2. For each game, it checks the corresponding metadata file to see if the game ended in a crash.
3. If a game crashed, the last 200 recorded actions (rows) are removed.
4. For all games, any action labeled as 'NOTHING' is removed.
5. All cleaned data is combined into a single dataset.
6. The key driving actions ('steer_left', 'steer_right', 'accelerate', 'decelerate')
   are oversampled to ensure each action has the same number of data points
   as the most frequent action among them.
7. The final, balanced dataset is shuffled and saved to a new CSV file.
"""

import os
import json
import pandas as pd
from typing import List

def process_and_combine_game_runs(
    source_dir: str = "data",
    output_file: str = "processed_training_data.csv"
) -> None:
    """
    Loads game data from CSV files, processes it, oversamples key actions,
    and saves it to a single combined CSV file.

    :param source_dir: Directory containing the raw CSV and metadata files.
    :param output_file: Path to save the final processed CSV file.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    processed_data_frames: List[pd.DataFrame] = []
    
    # Find all metadata files to identify the game runs
    metadata_files = [f for f in os.listdir(source_dir)
                      if f.startswith('game_') and f.endswith('_metadata.json')]
    
    print(f"Found {len(metadata_files)} game runs to process in '{source_dir}'.")

    for metadata_file in metadata_files:
        game_run_id = metadata_file.replace('_metadata.json', '')
        csv_file = f"{game_run_id}.csv"
        csv_path = os.path.join(source_dir, csv_file)
        metadata_path = os.path.join(source_dir, metadata_file)

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file '{csv_file}' not found. Skipping.")
            continue

        try:
            game_df = pd.read_csv(csv_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            is_crashed = metadata.get('crashed', False)

            # 1. If the game crashed, remove the last 200 actions
            if is_crashed and len(game_df) > 200:
                game_df = game_df.iloc[:-200]
            
            # 2. Remove all rows where the action is 'NOTHING'
            if 'action' in game_df.columns:
                game_df = game_df[game_df['action'] != 'NOTHING']

            if not game_df.empty:
                processed_data_frames.append(game_df)

        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

    if not processed_data_frames:
        print("No valid data was processed. The output file will not be created.")
        return

    # 3. Combine all processed data into a single DataFrame
    combined_df = pd.concat(processed_data_frames, ignore_index=True)
    print(f"\nCombined data from {len(processed_data_frames)} runs. Total rows before oversampling: {len(combined_df)}")
    
    if 'action' not in combined_df.columns:
        print("Warning: 'action' column not found in combined data. Skipping oversampling.")
        final_dataset = combined_df
    else:
        # 4. Perform oversampling
        print("Starting oversampling for key actions...")
        actions_to_balance = ['STEER_LEFT', 'STEER_RIGHT', 'ACCELERATE', 'DECELERATE']

        # Separate the data into parts to balance and parts to keep as is
        df_to_balance = combined_df[combined_df['action'].isin(actions_to_balance)]
        df_other = combined_df[~combined_df['action'].isin(actions_to_balance)]
        
        if df_to_balance.empty:
            print("No data found for the specified actions to balance. Skipping oversampling.")
            final_dataset = df_other
        else:
            action_counts = df_to_balance['action'].value_counts()
            max_size = action_counts.max()
            print(f"Target sample size for oversampling: {max_size} (from action '{action_counts.idxmax()}')")
            
            lst_balanced = [df_other]
            for action_name in actions_to_balance:
                # Check if the action exists in the data to avoid errors
                if action_name in action_counts:
                    action_df = df_to_balance[df_to_balance['action'] == action_name]
                    # Oversample with replacement
                    resampled_df = action_df.sample(n=max_size, replace=True, random_state=123)
                    lst_balanced.append(resampled_df)
                    print(f" - Resampled '{action_name}' from {len(action_df)} to {max_size} samples.")
            
            balanced_df = pd.concat(lst_balanced)
            
            # 5. Shuffle the final dataset to mix the oversampled rows
            print("\nShuffling final dataset...")
            final_dataset = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 6. Save the final dataset to a new CSV file
    final_dataset.to_csv(output_file, index=False)
    
    print("\nProcessing complete!")
    print(f"Total rows in the new balanced dataset: {len(final_dataset)}")
    print(f"Processed and balanced data has been saved to '{output_file}'")

def main():
    """
    Main function to run the data processing script.
    """
    print("Starting Race Car Data Processing Script")
    print("========================================")
    
    data_folder = "data"
    output_csv_file = "processed_balanced_training_data.csv"
    
    process_and_combine_game_runs(source_dir=data_folder, output_file=output_csv_file)
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()