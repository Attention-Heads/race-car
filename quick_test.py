#!/usr/bin/env python3
"""
Simple script to quickly test model predictions against expert data
Supports both PPO and Behavioral Cloning models
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from stable_baselines3 import PPO
from collections import Counter
from preprocessing_utils import StatePreprocessor
from bc_model_wrapper import load_bc_model

def load_model_and_scaler(model_type="ppo"):
    """Load the trained model and preprocessor."""
    try:
        if model_type.lower() == "bc":
            agent = load_bc_model("./models/best_bc_model.pth")
            print(f"✓ Behavioral Cloning model loaded successfully")
        else:
            agent = PPO.load("./models/best_model/best_model.zip")
            print(f"✓ PPO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load {model_type.upper()} model: {e}")
        return None, None
    
    try:
        preprocessor = StatePreprocessor(use_velocity_scaler=True)
        if preprocessor.velocity_scaler is not None:
            print("✓ Velocity scaler loaded successfully")
            print(f"  Scaler mean: {preprocessor.velocity_scaler.mean_}")
            print(f"  Scaler scale: {preprocessor.velocity_scaler.scale_}")
        else:
            print("⚠ WARNING: No velocity scaler found! This will cause preprocessing mismatch.")
            print("  The model was likely trained with velocity scaling, but scaler file is missing.")
            print("  This explains the poor accuracy - please retrain or regenerate the scaler.")
        print("✓ Preprocessor loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load preprocessor: {e}")
        preprocessor = StatePreprocessor(use_velocity_scaler=False)
        print("✓ Fallback preprocessor loaded (no velocity scaling)")
    
    return agent, preprocessor

def preprocess_state(row, preprocessor):
    """Convert a data row to model input format using centralized preprocessing."""
    return preprocessor.preprocess_dataframe_row(row)

def predict_action(agent, state_array):
    """Get action prediction from the model."""
    action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
    
    try:
        print(f"Predicting with state: {state_array}")
        action_num, _ = agent.predict(state_array, deterministic=True)
        return action_mapping.get(int(action_num), 'NOTHING')
    except Exception as e:
        print(f"Prediction error: {e}")
        return 'NOTHING'

def test_sample_accuracy(sample_size=1000, model_type="ppo"):
    """Test model accuracy on a sample of the expert data."""
    print(f"Loading {model_type.upper()} model and data...")
    
    # Load model
    agent, preprocessor = load_model_and_scaler(model_type)
    if agent is None or preprocessor is None:
        return
    
    # Load data
    try:
        df = pd.read_csv('processed_balanced_training_data.csv')
        print(f"✓ Loaded dataset with {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Sample data
    if sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Testing on random sample of {sample_size} rows")
    else:
        df_sample = df
        print(f"Testing on all {len(df)} rows")
    
    # Run predictions
    print("\nRunning predictions...")
    correct = 0
    total = 0
    predictions = []
    ground_truth = []
    
    for idx, row in df_sample.iterrows():
        if total % 100 == 0:
            print(f"Progress: {total}/{len(df_sample)}")
        
        # Get model prediction
        state_array = preprocess_state(row, preprocessor)
        predicted_action = predict_action(agent, state_array)
        true_action = row['action']
        
        predictions.append(predicted_action)
        ground_truth.append(true_action)
        
        if predicted_action == true_action:
            correct += 1
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n" + "="*50)
    print(f"RESULTS - {model_type.upper()} MODEL")
    print("="*50)
    print(f"Total samples tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show action distributions
    print(f"\nGround Truth Action Distribution:")
    gt_counts = Counter(ground_truth)
    for action, count in gt_counts.most_common():
        pct = (count / total) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    print(f"\nPredicted Action Distribution:")
    pred_counts = Counter(predictions)
    for action, count in pred_counts.most_common():
        pct = (count / total) * 100
        print(f"  {action}: {count} ({pct:.1f}%)")
    
    # Per-action accuracy
    print(f"\nPer-Action Accuracy:")
    for action in sorted(set(ground_truth)):
        action_mask = [gt == action for gt in ground_truth]
        action_predictions = [predictions[i] for i, mask in enumerate(action_mask) if mask]
        action_correct = sum(1 for pred in action_predictions if pred == action)
        action_total = len(action_predictions)
        action_acc = action_correct / action_total if action_total > 0 else 0
        print(f"  {action}: {action_acc:.4f} ({action_acc*100:.1f}%) - {action_correct}/{action_total}")
    
    # Save results
    results_df = pd.DataFrame({
        'true_action': ground_truth,
        'predicted_action': predictions,
        'correct': [t == p for t, p in zip(ground_truth, predictions)]
    })
    results_filename = f'quick_test_results_{model_type}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\nDetailed results saved to '{results_filename}'")
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model predictions against expert data')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Number of samples to test (default: 1000)')
    parser.add_argument('--model-type', type=str, choices=['ppo', 'bc'], default='ppo',
                       help='Model type to test: ppo or bc (default: ppo)')
    
    args = parser.parse_args()
    
    print(f"Testing {args.model_type.upper()} model with {args.sample_size} samples")
    test_sample_accuracy(args.sample_size, args.model_type)
