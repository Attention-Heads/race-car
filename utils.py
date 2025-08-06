import numpy as np

def print_game_state(game_state, max_sensor_range, sensor_names):
    """Formats the game state for printing."""
    s = game_state.sensors
    # Helper to format each sensor value
    def get_s(name):
        val = s.get(name, max_sensor_range)
        if val is None: # Handle potential None values from sensors
            val = max_sensor_range
        return f"{int(np.clip(val, 0, max_sensor_range)):04d}"

    # Create shorthands for all 16 sensors
    shorthand_map = {
        'left_side': 'LS', 'left_side_front': 'LSF', 'left_front': 'LF', 'front_left_front': 'FLF',
        'front': 'F', 'front_right_front': 'FRF', 'right_front': 'RF', 'right_side_front': 'RSF',
        'right_side': 'RS', 'right_side_back': 'RSB', 'right_back': 'RB', 'back_right_back': 'BRB',
        'back': 'B', 'back_left_back': 'BLB', 'left_back': 'LB', 'left_side_back': 'LSB'
    }

    sensor_strings = [f"{shorthand_map[name]}:{get_s(name)}" for name in sensor_names]
    compact_sensors = ' '.join(sensor_strings)

    print(
        f"D: {game_state.distance:<5} | "
        f"T: {game_state.elapsed_time_ms:<6} | "
        f"V:({game_state.velocity.get('x', 0):>2},{game_state.velocity.get('y', 0):>2}) | "
        f"C: {'Y' if game_state.did_crash else 'N'} | "
        f"S:[{compact_sensors}]"
    )
