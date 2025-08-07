# PPO Fine-tuning Script Usage Examples

This document shows how to use the improved `ppo_finetune.py` script with various modes and parameters.

## Key Improvements

1. **Best Model Tracking**: Automatically saves the best model based on distance traveled
2. **Resume Training**: Continue training from saved checkpoints
3. **Optimized Hyperparameters**: Better default values for fine-tuning
4. **Comprehensive Evaluation**: Detailed evaluation with statistics
5. **Checkpointing**: Regular model saves during training
6. **Metadata Tracking**: JSON metadata files with training progress

## Usage Examples

### 1. Train from Evolutionary Model (New Training)
```bash
python ppo_finetune.py --model-path evolutionary_results/best_model.pkl --mode train --timesteps 500000 --learning-rate 3e-4 --n-envs 8
```

### 2. Resume Training from PPO Checkpoint
```bash
python ppo_finetune.py --model-path checkpoints/ppo_checkpoint_100000_steps.zip --mode resume --resume-timesteps 300000
```

### 3. Fine-tune Existing PPO Model
```bash
python ppo_finetune.py --model-path existing_ppo_model.zip --mode train --timesteps 200000
```

### 4. Evaluate a Trained Model
```bash
python ppo_finetune.py --model-path best_model.zip --mode eval --eval-episodes 20 --render
```

### 5. Custom Hyperparameters Training
```bash
python ppo_finetune.py --model-path evolutionary_results/best_model.pkl --mode train --timesteps 1000000 --learning-rate 2e-4 --batch-size 512 --n-epochs 15 --clip-range 0.3 --ent-coef 0.02 --n-envs 12
```

## Important Files Generated

- `best_model.zip` - The best performing model (based on distance)
- `best_model_metadata.json` - Metadata about the best model
- `ppo_finetuned_model.zip` - Final trained model
- `checkpoints/` - Regular training checkpoints
- `logs/` - Tensorboard logs

## Key Parameters

### Training Parameters
- `--timesteps`: Total training timesteps (default: 500,000)
- `--learning-rate`: Learning rate (default: 3e-4, optimized for fine-tuning)
- `--batch-size`: Batch size (default: 256)
- `--n-epochs`: Update epochs (default: 10)
- `--clip-range`: PPO clip range (default: 0.2)
- `--n-envs`: Parallel environments (default: 8, reduced for stability)

### Monitoring Parameters
- `--checkpoint-freq`: Checkpoint frequency in timesteps (default: 50,000)
- `--eval-freq`: Evaluation frequency in timesteps (default: 25,000)
- `--eval-episodes`: Episodes per evaluation (default: 10)

### File Parameters
- `--output-path`: Final model save path
- `--best-model-path`: Best model save path (default: 'best_model.zip')

## Best Practices

1. **Start Small**: Begin with 500K timesteps, then increase if needed
2. **Monitor Progress**: Check `best_model_metadata.json` for progress
3. **Use Resume**: If training is interrupted, resume from checkpoints
4. **Evaluate Regularly**: Use eval mode to test model performance
5. **Adjust Environments**: Reduce `--n-envs` if you have memory issues
6. **Focus on Distance**: The reward is optimized for distance traveled

## Reward Configuration

The script uses optimized reward settings:
- Distance progress: 1.0 (primary signal)
- Crash penalty: -50.0 (moderate, not overwhelming)
- Time penalty: -0.01 (encourages faster completion)
- Speed bonus: 0.0 (disabled to focus purely on distance)

## Monitoring Training

1. **Tensorboard**: `tensorboard --logdir logs/`
2. **Best Model Metadata**: Check `best_model_metadata.json`
3. **Console Output**: Real-time progress and evaluation results
4. **Checkpoints**: Regular saves in `checkpoints/` directory
