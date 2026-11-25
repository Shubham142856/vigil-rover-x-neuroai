"""Provisional classical stopping rule for the M1 comparator."""
import math


def decide(observation):
    """Return a pilot action and its reason from one delivered observation."""
    left = observation.get("whisker_left")
    right = observation.get("whisker_right")
    if left or right:
        return "brake", "tactile contact"
    if left is None or right is None:
        return "brake", "missing tactile observation"
    if observation.get("frame_valid") != True:
        return "brake", "invalid camera frame"

    fields = ("frame_age_s", "brightness", "contrast", "obstacle_center", "looming_center")
    values = [observation.get(key) for key in fields]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        return "brake", "incomplete camera observation"
    age, brightness, contrast, center, growth = values
    if not 0 <= age <= 0.2:
        return "brake", "stale camera frame"
    if not (0 <= brightness <= 1 and 0 <= contrast <= 1 and 0 <= center <= 1):
        return "brake", "invalid image features"
    if (brightness > 0.92 or brightness < 0.03) and contrast < 0.06:
        return "brake", "insufficient camera contrast"
    if center > 0.30 or growth > 0.08:
        return "brake", "frontal visual warning"
    return "forward", "below pilot warning thresholds"
