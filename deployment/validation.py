from __future__ import annotations

import math
from datetime import datetime
from urllib.parse import urlparse


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
REQUIRED_KEYS = {
    "request_id",
    "pollutant",
    "concentration_unit",
    "coordinate_system",
    "stations",
    "wind",
    "concentrations",
    "callback_url",
}


class InputValidationError(ValueError):
    pass


def parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise InputValidationError(f"{field} must be a string in {TIME_FORMAT} format")
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError as exc:
        raise InputValidationError(
            f"{field} must use YYYY-MM-DD HH:MM:SS format"
        ) from exc


def finite_number(value: object, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise InputValidationError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise InputValidationError(f"{field} must be greater than or equal to {minimum}")
    return number


def nonblank(value: object, field: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise InputValidationError(f"{field} cannot be blank")
    return text


def validate_input(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise InputValidationError("request body must be a JSON object")
    missing = sorted(REQUIRED_KEYS - payload.keys())
    if missing:
        raise InputValidationError(f"missing required fields: {', '.join(missing)}")

    normalized = dict(payload)
    normalized["request_id"] = nonblank(payload["request_id"], "request_id")
    normalized["pollutant"] = nonblank(payload["pollutant"], "pollutant")
    normalized["concentration_unit"] = nonblank(
        payload["concentration_unit"], "concentration_unit"
    )
    if payload["coordinate_system"] != "WGS84":
        raise InputValidationError("coordinate_system must be WGS84")

    callback = nonblank(payload["callback_url"], "callback_url")
    parsed_callback = urlparse(callback)
    if parsed_callback.scheme not in {"http", "https"} or not parsed_callback.netloc:
        raise InputValidationError("callback_url must be an HTTP or HTTPS URL")

    stations = payload["stations"]
    if not isinstance(stations, list) or not stations:
        raise InputValidationError("stations must contain at least one station")
    station_ids: set[str] = set()
    normalized_stations = []
    for index, station in enumerate(stations):
        if not isinstance(station, dict):
            raise InputValidationError(f"stations[{index}] must be an object")
        station_id = nonblank(station.get("station_id"), f"stations[{index}].station_id")
        if station_id in station_ids:
            raise InputValidationError(f"duplicate station_id: {station_id}")
        station_ids.add(station_id)
        lon = finite_number(station.get("longitude"), f"stations[{index}].longitude")
        lat = finite_number(station.get("latitude"), f"stations[{index}].latitude")
        if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
            raise InputValidationError(f"stations[{index}] coordinate is outside WGS84 range")
        normalized_stations.append(
            {"station_id": station_id, "longitude": lon, "latitude": lat}
        )

    wind = payload["wind"]
    if not isinstance(wind, list) or not wind:
        raise InputValidationError("wind must contain at least one record")
    wind_times: set[datetime] = set()
    normalized_wind = []
    for index, record in enumerate(wind):
        if not isinstance(record, dict):
            raise InputValidationError(f"wind[{index}] must be an object")
        time = parse_time(record.get("time"), f"wind[{index}].time")
        if time in wind_times:
            raise InputValidationError(f"duplicate wind time: {record.get('time')}")
        wind_times.add(time)
        speed = finite_number(record.get("sp"), f"wind[{index}].sp", minimum=0.0)
        direction = finite_number(record.get("dir"), f"wind[{index}].dir")
        if not 0.0 <= direction < 360.0:
            raise InputValidationError(f"wind[{index}].dir must be in [0, 360)")
        normalized_wind.append({"time": time, "sp": speed, "dir": direction})

    concentrations = payload["concentrations"]
    if not isinstance(concentrations, list) or not concentrations:
        raise InputValidationError("concentrations must contain at least one record")
    concentration_keys: set[tuple[datetime, str]] = set()
    concentration_times: set[datetime] = set()
    normalized_concentrations = []
    for index, record in enumerate(concentrations):
        if not isinstance(record, dict):
            raise InputValidationError(f"concentrations[{index}] must be an object")
        time = parse_time(record.get("time"), f"concentrations[{index}].time")
        station_id = nonblank(
            record.get("station_id"), f"concentrations[{index}].station_id"
        )
        if station_id not in station_ids:
            raise InputValidationError(
                f"concentrations[{index}].station_id does not exist: {station_id}"
            )
        key = (time, station_id)
        if key in concentration_keys:
            raise InputValidationError(
                f"duplicate concentration record: {time.strftime(TIME_FORMAT)}, {station_id}"
            )
        concentration_keys.add(key)
        concentration_times.add(time)
        value = finite_number(
            record.get("value"), f"concentrations[{index}].value", minimum=0.0
        )
        normalized_concentrations.append(
            {"time": time, "station_id": station_id, "value": value}
        )

    common_times = wind_times & concentration_times
    if not common_times:
        raise InputValidationError("wind and concentrations have no common timestamps")

    normalized["stations"] = normalized_stations
    normalized["wind"] = sorted(normalized_wind, key=lambda row: row["time"])
    normalized["concentrations"] = sorted(
        normalized_concentrations, key=lambda row: (row["time"], row["station_id"])
    )
    return normalized
