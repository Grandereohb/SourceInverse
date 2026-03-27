import math
import re
import numpy as np


def dms_to_decimal(dms_str: str) -> float:
    """Convert DMS to decimal degrees."""
    if dms_str is None:
        raise ValueError("Invalid DMS format: None")

    # If already numeric, return directly
    if isinstance(dms_str, (int, float, np.floating)):
        if np.isnan(dms_str):
            raise ValueError("Invalid DMS format: NaN")
        return float(dms_str)

    s = str(dms_str).strip()

    # If it's already a plain decimal string, return directly
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        return float(s)

    # Normalize degree symbol variants / mojibake
    deg_mojibake_1 = chr(0x00C2) + chr(0x00B0)  # ??
    deg_mojibake_2 = chr(0x040E) + chr(0x0433)  # ??
    deg_symbol = chr(0x00B0)
    s = s.replace(deg_mojibake_1, deg_symbol)
    s = s.replace(deg_mojibake_2, deg_symbol)
    for deg_sym in [chr(0x63B3), chr(0x00BA), chr(0x02DA), deg_symbol]:
        s = s.replace(deg_sym, deg_symbol)

    s = s.replace(chr(0x2032), "'").replace(chr(0x2033), '"')

    # Normalize full-width digits and symbols
    full = [chr(c) for c in range(0xFF10, 0xFF1A)] + [chr(0xFF0E), chr(0xFF0D), chr(0xFF0B)]
    half = list("0123456789.-+")
    trans = {full[i]: half[i] for i in range(len(half))}
    s = s.translate(str.maketrans(trans))

    # Extract numbers robustly
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", s)]
    if len(nums) >= 3:
        deg, minute, sec = nums[0], nums[1], nums[2]
        return deg + minute / 60.0 + sec / 3600.0
    if len(nums) == 2:
        deg, minute = nums[0], nums[1]
        return deg + minute / 60.0
    if len(nums) == 1:
        return nums[0]
    raise ValueError(f"Invalid DMS format: {dms_str}")


def latlon_to_xy(lon, lat, lon0, lat0):
    """Local tangent plane in meters."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


def xy_to_latlon(x, y, lon0, lat0):
    """Inverse of local tangent plane projection."""
    lon = lon0 + x / (math.cos(math.radians(lat0)) * 111320.0)
    lat = lat0 + y / 110540.0
    return lon, lat
