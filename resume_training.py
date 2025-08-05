#!/usr/bin/env python3
"""
Helper script to continue evolutionary training from existing results
"""

import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Resume evolutionary training from existing results")
    parser.add_argument("--results-dir", type=str, default="evolutionary_results", 
                       help="Directory containing evolutionary results")
    parser.add_argument("--generations", type=int, default=200, 
                       help="Total generations to train (including resumed)")
    parser.add_argument("--population-size", type=int, default=100,
                       help="Population size")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    
    # Check for existing population checkpoints
    checkpoint_files = list(results_path.glob("population_checkpoint_gen_*.pkl"))
    
    if checkpoint_files:
        # Sort by generation number and use the latest
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = str(checkpoint_files[-1])
        print(f"Found existing population checkpoint: {latest_checkpoint}")
        
        # Resume from checkpoint
        cmd = f'python train_evolutionary.py --resume "{latest_checkpoint}" --generations {args.generations} --population-size {args.population_size}'
        if args.workers:
            cmd += f' --workers {args.workers}'
        
        print(f"Resuming training with command:")
        print(cmd)
        os.system(cmd)
        
    else:
        # No population checkpoints found, try to create one from best individual
        best_individual_files = list(results_path.glob("best_individual_*.pkl"))
        
        if best_individual_files:
            # Use the final one if available, otherwise the latest generation
            final_file = results_path / "best_individual_final.pkl"
            if final_file.exists():
                best_file = str(final_file)
            else:
                # Sort by generation number and use the latest
                gen_files = [f for f in best_individual_files if "gen_" in f.name]
                if gen_files:
                    gen_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    best_file = str(gen_files[-1])
                else:
                    best_file = str(best_individual_files[0])
            
            print(f"Found best individual file: {best_file}")
            print("Creating population checkpoint from best individual...")
            
            # Create checkpoint
            checkpoint_path = results_path / "population_checkpoint_from_best.pkl"
            cmd1 = f'python train_evolutionary.py --create-checkpoint "{best_file}" --checkpoint-output "{checkpoint_path}" --population-size {args.population_size}'
            
            print(f"Creating checkpoint with command:")
            print(cmd1)
            os.system(cmd1)
            
            # Resume training
            cmd2 = f'python train_evolutionary.py --resume "{checkpoint_path}" --generations {args.generations} --population-size {args.population_size}'
            if args.workers:
                cmd2 += f' --workers {args.workers}'
            
            print(f"Resuming training with command:")
            print(cmd2)
            os.system(cmd2)
            
        else:
            print(f"No evolutionary results found in {args.results_dir}")
            print("Please run initial training first with: python train_evolutionary.py")

if __name__ == "__main__":
    main()
