import numpy as np
import pandas as pd

from geo_utils import dms_to_decimal, latlon_to_xy

TARGET_POLLUTANT_COLUMN = "TARGET_POLLUTANT"


def load_sites(path):
    df = pd.read_excel(path)
    # Two supported formats:
    # A) columns: station, lon, lat
    # B) columns: station, N, S, E ... with rows [lon, lat]
    cols = {str(c).lower(): c for c in df.columns}

    if "station" in cols and any(
        str(v).lower() in ["lon", "lat"] for v in df[cols["station"]].tolist()
    ):
        # Format B
        st_col = cols["station"]
        station_cols = [c for c in df.columns if c != st_col]
        df2 = df.set_index(st_col)
        lons = [dms_to_decimal(df2.loc["lon", c]) for c in station_cols]
        lats = [dms_to_decimal(df2.loc["lat", c]) for c in station_cols]
        stations = [str(c) for c in station_cols]
    else:
        # Format A
        st_col = cols.get("station", df.columns[0])
        lon_col = cols.get("lon", df.columns[1])
        lat_col = cols.get("lat", df.columns[2])
        stations = df[st_col].astype(str).tolist()
        lons = [dms_to_decimal(v) for v in df[lon_col].tolist()]
        lats = [dms_to_decimal(v) for v in df[lat_col].tolist()]

    lon0 = float(np.mean(lons))
    lat0 = float(np.mean(lats))
    xy = [latlon_to_xy(lon, lat, lon0, lat0) for lon, lat in zip(lons, lats)]
    site_df = pd.DataFrame(
        {
            "station": stations,
            "lon": lons,
            "lat": lats,
            "x": [p[0] for p in xy],
            "y": [p[1] for p in xy],
        }
    )
    return site_df, lon0, lat0


def load_wind(path):
    df = pd.read_excel(path)
    # Expect columns: time, dir, sp (first column is time)
    cols = {str(c).lower(): c for c in df.columns}
    t_col = df.columns[0]
    dir_col = cols.get("dir", df.columns[1])
    sp_col = cols.get("sp", df.columns[2])
    out = df[[t_col, dir_col, sp_col]].copy()
    out.columns = ["time", "dir", "sp"]
    out["time"] = pd.to_datetime(out["time"])
    return out


def load_conc(path):
    df = pd.read_excel(path)
    # Expect columns: time, N, E, S (or other station labels); first column is time
    t_col = df.columns[0]
    out = df.copy()
    out = out.rename(columns={t_col: "time"})
    out["time"] = pd.to_datetime(out["time"])
    if TARGET_POLLUTANT_COLUMN in out.columns:
        values = out[TARGET_POLLUTANT_COLUMN].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            out.attrs["target_pollutant"] = values.iloc[0]
        out = out.drop(columns=[TARGET_POLLUTANT_COLUMN])
    return out


def get_concentration_target_pollutant(conc_df):
    value = conc_df.attrs.get("target_pollutant")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def wind_dir_to_uv(dir_deg, sp, is_from=True):
    # dir_deg: 0 = North, 90 = East, clockwise
    rad = np.deg2rad(dir_deg)
    if is_from:
        u = -sp * np.sin(rad)
        v = -sp * np.cos(rad)
    else:
        u = sp * np.sin(rad)
        v = sp * np.cos(rad)
    return u, v
