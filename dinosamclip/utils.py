import random


def get_color():
    """Generate a bright, high-contrast color using HSV color space."""
    hue = random.random()  # Random hue (0-1)
    saturation = 0.8 + random.random() * 0.2  # High saturation (0.8-1.0)
    value = 0.8 + random.random() * 0.2  # High brightness (0.8-1.0)

    # Convert HSV to RGB (assuming values in range 0-1)
    h = hue * 6
    i = int(h)
    f = h - i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)

    if i == 0:
        return [value, t, p]
    elif i == 1:
        return [q, value, p]
    elif i == 2:
        return [p, value, t]
    elif i == 3:
        return [p, q, value]
    elif i == 4:
        return [t, p, value]
    else:
        return [value, p, q]
