#!/usr/bin/env python3
"""
Simple script to quickly test model predictions against expert data
"""

import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from collections import Counter
from preprocessing_utils import StatePreprocessor

def load_model_and_scaler():
    """Load the trained model and preprocessor."""
    try:
        agent = PPO.load("./models/final_model.zip")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None
    
    try:
        preprocessor = StatePreprocessor(use_velocity_scaler=True)
        print("✓ Preprocessor loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load preprocessor: {e}")
        preprocessor = StatePreprocessor(use_velocity_scaler=False)
    
    return agent, preprocessor

def preprocess_state(row, preprocessor):
    """Convert a data row to model input format using centralized preprocessing."""
    return preprocessor.preprocess_dataframe_row(row)

def predict_action(agent, state_array):
    """Get action prediction from the model."""
    action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
    
    try:
        action_num, _ = agent.predict(state_array, deterministic=True)
        return action_mapping.get(int(action_num), 'NOTHING')
    except Exception as e:
        print(f"Prediction error: {e}")
        return 'NOTHING'

def test_sample_accuracy(sample_size=1000):
    """Test model accuracy on a sample of the expert data."""
    print("Loading model and data...")
    
    # Load model
    agent, preprocessor = load_model_and_scaler()
    if agent is None or preprocessor is None:
        return
    
    # Load data
    try:
        df = pd.read_csv('expert_training_data.csv')
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
    print("RESULTS")
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
    results_df.to_csv('quick_test_results.csv', index=False)
    print(f"\nDetailed results saved to 'quick_test_results.csv'")
    
    return accuracy

if __name__ == "__main__":
    import sys
    
    # Default sample size
    sample_size = 1000
    
    # Check for command line argument
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
        except ValueError:
            print("Usage: python quick_test.py [sample_size]")
            sys.exit(1)
    
    test_sample_accuracy(sample_size)
