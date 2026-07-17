SECONDS_PER_HOUR = 3600.0


def normalize_velocity_mps(velocity_mps, duration_hours, length_m, factor=1.0):
    """Convert m/s velocity to normalized distance per normalized time."""
    length_m = max(float(length_m), 1e-12)
    return (
        velocity_mps
        * SECONDS_PER_HOUR
        * float(duration_hours)
        / length_m
        * float(factor)
    )


def normalize_diffusivity_m2s(diffusivity_m2s, duration_hours, length_m):
    """Convert m^2/s diffusivity to normalized distance^2 per normalized time."""
    length_m = max(float(length_m), 1e-12)
    return (
        diffusivity_m2s
        * SECONDS_PER_HOUR
        * float(duration_hours)
        / (length_m**2)
    )


def normalize_decay_per_hour(decay_per_hour, duration_hours):
    """Convert a first-order hourly decay rate to normalized inverse time."""
    return float(decay_per_hour) * float(duration_hours)
