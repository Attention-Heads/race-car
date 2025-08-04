# PPO Model Demo - Watch Your AI Play!

This guide shows you how to use your trained PPO model (initialized with behavioral cloning) to play the race car game so you can watch your AI in action.

## Setup Overview

The setup consists of:
1. **API Server** (`api.py`) - Serves your trained PPO model
2. **Game Client** (`example.py`) - The race car game that calls your API
3. **Demo Scripts** - Easy-to-use scripts to run everything

## Quick Start

### Option 1: Use the Demo Script (Recommended)
```bash
python run_ppo_demo.py
```

This will:
- Automatically start the API server with your PPO model
- Launch the game where your AI controls the car
- Clean up everything when you're done

### Option 2: Manual Setup
If you prefer to run things separately:

1. **Start the API server:**
```bash
python api.py
```

2. **In a new terminal, run the game:**
```bash
python example.py
```

## Testing Your Setup

Before running the full demo, you can test if everything is working:

```bash
python test_api.py
```

This will verify that:
- Your PPO model loads correctly
- The API responds to requests
- Valid actions are returned

## What You'll See

When you run the demo:
1. A pygame window will open showing the race car game
2. Your PPO model will control the yellow car
3. You can watch how well your AI drives!
4. The game will print actions and game state to the console

## Files Modified

The following files have been updated to support API-based gameplay:

- `api.py` - Updated to load your `ppo_initialized_with_bc.zip` model
- `src/game/core.py` - Added `get_action_from_api()` function
- `example.py` - Configured to use API instead of keyboard input

## Model Details

- **Model**: `models/ppo_initialized_with_bc.zip`
- **Scaler**: `velocity_scaler.pkl`
- **API Endpoint**: `http://localhost:9052/predict`

## Controls

- The AI controls the car automatically
- Close the game window or press ESC to exit
- The game runs at 60 FPS

## Troubleshooting

### "Model not loaded" error
- Check that `models/ppo_initialized_with_bc.zip` exists
- Check that `velocity_scaler.pkl` exists
- Check the console output when starting the API server

### "Could not connect to API" error
- Make sure the API server is running first
- Check that nothing else is using port 9052
- Try running `python test_api.py` to diagnose the issue

### Game crashes or freezes
- Make sure you have pygame installed: `pip install pygame`
- Check the console output for error messages
- Try running with a different seed value in `example.py`

## Customization

You can modify the behavior by:

- **Changing the seed**: Edit `seed_value` in `example.py` or `run_ppo_demo.py`
- **Using different models**: Change the model path in `api.py`
- **Adjusting game settings**: Modify constants in `src/game/core.py`

## Performance Tips

- The AI makes decisions at 60 FPS, so API calls need to be fast
- If the game is slow, check your model inference time
- You can adjust the timeout in `get_action_from_api()` if needed

Enjoy watching your AI drive! 🏁🤖
