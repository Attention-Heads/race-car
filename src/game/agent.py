from typing import List

# Define constants for better readability and easier tuning
WALL_CRITICAL_DIST = 70  # Very close to wall, aggressive action needed
WALL_DANGER_DIST = 150   # Approaching wall, proactive action needed

FRONT_CRITICAL_DIST = 300  # Imminent frontal collision, requires deceleration
FRONT_DANGER_DIST = 600   # Close frontal obstacle, now focuses on steering
FRONT_WARNING_DIST = 800  # Obstacle at a distance, proactive steering
FRONT_VERY_CLEAR_DIST = 1000  # Distance required for safe acceleration

BACK_DANGER_DIST = 400    # Car dangerously close from behind
BACK_WARNING_DIST = 800   # Car approaching from behind

# Global variable to store the last action, as per original code structure
last_action = ["NOTHING"]


def get_action_from_rule_based_agent(sensor_data: dict) -> List[str]:
    """
    Determines the car's actions based on a complex rule-based approach.
    This version has refined steering logic to prevent overcorrection and
    reduce crashes into walls, while maintaining less acceleration and deceleration.

    Args:
        sensor_data (dict): A dictionary containing sensor readings from 16 sensors.
                            Keys are sensor names (e.g., "front", "left_side"),
                            values are distances in pixels or None if no obstacle detected.

    Returns:
        List[str]: A list of actions for the car (e.g., ["ACCELERATE", "STEER_LEFT"]).
    """

    def sanitize(value):
        """
        Replaces None sensor readings with a default maximum range (1000px).
        This treats out-of-range obstacles as being at the maximum perceived distance.
        """
        return value if value is not None else 1000

    # Sanitize and extract all 16 sensor readings for comprehensive use
    s = {name: sanitize(sensor_data.get(name)) for name in [
        "left_side", "left_side_front", "left_front", "front_left_front",
        "front", "front_right_front", "right_front", "right_side_front",
        "right_side", "right_side_back", "right_back", "back_right_back",
        "back", "back_left_back", "left_back", "left_side_back"
    ]}

    actions = []

    # --- 1. Wall Avoidance (Highest Priority) ---
    # Prioritize avoiding walls. Deceleration is now only a last resort.
    # Removed the comparison to the other side for critical wall avoidance
    if s["left_side"] < WALL_CRITICAL_DIST:
        actions.extend(["STEER_RIGHT"])
        if s["front"] < FRONT_CRITICAL_DIST or s["front_left_front"] < FRONT_CRITICAL_DIST:
            actions.insert(0, "DECELERATE")
    elif s["right_side"] < WALL_CRITICAL_DIST:
        actions.extend(["STEER_LEFT"])
        if s["front"] < FRONT_CRITICAL_DIST or s["front_right_front"] < FRONT_CRITICAL_DIST:
            actions.insert(0, "DECELERATE")
    # Reduced aggression for danger zone wall avoidance
    elif s["left_side"] < WALL_DANGER_DIST:
        actions.append("STEER_RIGHT")
    elif s["right_side"] < WALL_DANGER_DIST:
        actions.append("STEER_LEFT")

    if actions:
        global last_action
        last_action = actions
        return actions

    # --- 2. Frontal Obstacle Avoidance (High Priority) ---
    # Deceleration is now reserved only for critical frontal threats.
    left_path_threat = min(s["front_left_front"],
                           s["left_front"], s["left_side_front"])
    right_path_threat = min(s["front_right_front"],
                            s["right_front"], s["right_side_front"])

    # Critical frontal obstacle: immediate deceleration and aggressive steering
    if s["front"] < FRONT_CRITICAL_DIST:
        actions.append("DECELERATE")
        if left_path_threat > right_path_threat:
            actions.extend(["STEER_LEFT"])
        else:
            actions.extend(["STEER_RIGHT"])

    # Danger frontal obstacle: now only steers proactively (less aggressive steering)
    elif s["front"] < FRONT_DANGER_DIST:
        if left_path_threat > right_path_threat + 100:
            actions.append("STEER_LEFT")
        elif right_path_threat > left_path_threat + 100:
            actions.append("STEER_RIGHT")
        elif left_path_threat > FRONT_DANGER_DIST or right_path_threat > FRONT_DANGER_DIST:
            if left_path_threat > right_path_threat:
                actions.append("STEER_LEFT")
            else:
                actions.append("STEER_RIGHT")

    # Warning frontal obstacle: proactive steering to avoid future danger (single steer)
    elif s["front"] < FRONT_WARNING_DIST:
        if s["front_left_front"] < FRONT_WARNING_DIST and s["front_right_front"] >= FRONT_WARNING_DIST:
            actions.append("STEER_RIGHT")
        elif s["front_right_front"] < FRONT_WARNING_DIST and s["front_left_front"] >= FRONT_WARNING_DIST:
            actions.append("STEER_LEFT")
        elif s["front_left_front"] < FRONT_WARNING_DIST and s["front_right_front"] < FRONT_WARNING_DIST:
            if left_path_threat > right_path_threat:
                actions.append("STEER_LEFT")
            else:
                actions.append("STEER_RIGHT")

    if actions:
        last_action = actions
        return actions

    # --- 3. Rear Threat Management (Medium Priority) ---
    # Combine all rear sensors for a comprehensive threat assessment.
    rear_threat = min(s["back"], s["back_left_back"], s["left_back"], s["left_side_back"],
                      s["back_right_back"], s["right_back"], s["right_side_back"])

    # Check if the front path is clear enough to accelerate.
    front_clear_for_accel = (s["front"] > FRONT_WARNING_DIST and
                             left_path_threat > FRONT_WARNING_DIST and
                             right_path_threat > FRONT_WARNING_DIST)

    # Only accelerate moderately if a car is dangerously close from behind AND the front is clear.
    if rear_threat < BACK_DANGER_DIST and front_clear_for_accel:
        actions.append("ACCELERATE")
        if s["left_side"] < WALL_DANGER_DIST:
            actions.append("STEER_RIGHT")
        elif s["right_side"] < WALL_DANGER_DIST:
            actions.append("STEER_LEFT")

    if actions:
        last_action = actions
        return actions

    # --- 4. Lane Centering & Progress (Lowest Priority) ---
    # This logic only runs if no other threats are detected.

    # Use the left_front and right_front sensors to try and stay in the middle of the lane.
    lane_diff = s["left_front"] - s["right_front"]

    if abs(lane_diff) > 50:  # A noticeable difference in lane positioning
        if lane_diff < 0:  # Closer to the right side
            actions.append("STEER_LEFT")
        else:  # Closer to the left side
            actions.append("STEER_RIGHT")

    # Accelerate only if the path is very clear.
    if (s["front"] > FRONT_VERY_CLEAR_DIST and
        left_path_threat > FRONT_VERY_CLEAR_DIST - 100 and
            right_path_threat > FRONT_VERY_CLEAR_DIST - 100):
        actions.append("ACCELERATE")

    # If no other action has been taken, default to NOTHING to maintain current speed.
    if not actions:
        actions.append("NOTHING")

    # Update last_action for the next tick
    last_action = actions
    return actions
