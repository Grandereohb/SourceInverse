# Source Inversion HTTP API Protocol v1

This document defines job submission, status query, and result callback. JSON
uses UTF-8. See `input.json` and `output.json` for complete business examples.

## Common Rules

- Time: `YYYY-MM-DD HH:MM:SS`.
- Coordinate system: `WGS84`.
- Wind speed: `sp`, in `m/s`.
- Wind direction: `dir`, in `[0, 360)`, using meteorological wind-from degrees
  clockwise from north.
- Concentrations and coordinates are JSON numbers. Zero concentration is valid;
  missing data must not be replaced by zero.
- Production callbacks should use gzip. The initial body-size limit is 10 MB.

## Submit Job

`POST /api/v1/jobs`

Headers:

```text
Content-Type: application/json; charset=utf-8
Authorization: Bearer <token>
Idempotency-Key: <request_id>
```

Body: `input.json`.

The service responds immediately with HTTP `202 Accepted`:

```json
{
  "request_id": "client-20260811-001",
  "job_id": "a816c3e2-0f70-4b68-a9a0-cc6b0444b80d",
  "status": "queued"
}
```

`request_id` is unique. Repeating the same request returns the existing job;
reusing it with different data returns HTTP `409`.

## Query Status

`GET /api/v1/jobs/{job_id}`

```json
{
  "request_id": "client-20260811-001",
  "job_id": "a816c3e2-0f70-4b68-a9a0-cc6b0444b80d",
  "status": "running",
  "error": null
}
```

Statuses: `queued`, `validating`, `running`, `packaging`, `callback_pending`,
`completed`, and `failed`.

## Download Result

- `GET /api/v1/jobs/{job_id}/output` returns the machine-readable `output.json`.
- `GET /api/v1/jobs/{job_id}/result` returns the complete `result.zip` package.
- Both endpoints return HTTP `409` while their artifact is not ready.

## Result Callback

After training, the service sends `POST {callback_url}` with the body in
`output.json`.

```text
Content-Type: application/json; charset=utf-8
Content-Encoding: gzip
X-Request-ID: <request_id>
X-Job-ID: <job_id>
Authorization: Bearer <callback token>
```

Any HTTP `2xx` response confirms receipt. Otherwise the service retries.
Callbacks are idempotent by `job_id`.

Every concentration point explicitly contains `longitude`, `latitude`, and
`concentration`. The current `36 x 36` grid has 1,296 points per hourly field,
but clients should parse the list without assuming a permanent grid size.

`release_strength.value` is currently the learned model Q value with unit
`model_unit`, not a calibrated physical mass flow such as `kg/h` or `g/s`.

## Validation And Errors

- IDs and pollutant/unit names cannot be blank.
- Station IDs must be unique and referenced by concentration records.
- Longitude: `[-180, 180]`; latitude: `[-90, 90]`.
- `sp` and concentration must be finite and non-negative.
- `dir` must be finite and in `[0, 360)`.
- Duplicate wind times and duplicate `(time, station_id)` concentrations are
  rejected.

Errors use this structure:

```json
{
  "request_id": "client-20260811-001",
  "job_id": null,
  "error": {
    "code": "INVALID_WIND_DIRECTION",
    "message": "wind[0].dir must be greater than or equal to 0 and less than 360",
    "field": "wind[0].dir",
    "retryable": false
  }
}
```

Main HTTP codes: `400` invalid JSON, `401` authentication, `409` duplicate
conflict, `413` body too large, `422` invalid data, `429` queue full, and `500`
or `503` service failure.

## Deployment Configuration And Remaining Decisions

Bearer authentication, callback tokens, callback timeout/retry intervals, and
callback host restrictions are server environment settings. The server can
either force one callback URL or allow request URLs only for configured hosts.

The following business rules still require client confirmation:

1. Minimum station count, timestamp count, and common data duration.
2. Whether non-hourly input is rejected or resampled.
3. Allowed concentration units and numeric precision.
4. Whether `quality` is required in the callback.
5. Result retention period and automatic cleanup policy.
