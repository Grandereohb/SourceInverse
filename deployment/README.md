# Local Deployment Adapter

Run one isolated source-inversion job without HTTP:

```bash
python -m deployment.local_runner \
  --input proto/input.json \
  --work-root deployment_runs
```

The adapter validates the JSON request, writes the three Excel inputs expected
by the existing algorithm, runs `pipeline.run()` in a CPU child process, builds
`output.json`, writes a SHA-256 manifest, and packages the full job as
`result.zip`.

For a quick integration test, override training epochs without changing
`pinn_source/config.py`:

```bash
python -m deployment.local_runner \
  --input path/to/real_input.json \
  --work-root deployment_runs \
  --epochs 2
```

Omit `--epochs` for the configured production training length. Plot generation
is disabled by default for CPU deployment; pass `--make-plots` when required.
The epoch override is intended only for integration testing and must not be used
for production inversion results.

Each job is stored under `<work-root>/<request-id>_<job-id-prefix>/` with input,
algorithm output, delivery files, logs, and `job.json` status metadata.

Run adapter tests with:

```bash
python -m unittest discover -s deployment/tests -v
```

## HTTP Service

Install the API dependencies into the active environment, then start one
Uvicorn worker from the repository root:

```bash
pip install -r deployment/requirements-api.txt
export SOURCE_INVERSION_API_TOKEN='replace-with-a-secret'
export SOURCE_INVERSION_CALLBACK_TOKEN='replace-with-a-secret'
export SOURCE_INVERSION_WORK_ROOT='./deployment_runs/api'
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Submit and inspect an asynchronous job:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -H 'Idempotency-Key: client-20260811-001' \
  --data-binary @proto/input.json

curl -H 'Authorization: Bearer replace-with-a-secret' \
  http://127.0.0.1:8000/api/v1/jobs/<job-id>

curl -H 'Authorization: Bearer replace-with-a-secret' \
  -o result.zip http://127.0.0.1:8000/api/v1/jobs/<job-id>/result
```

The service also exposes `/api/v1/health` and
`/api/v1/jobs/<job-id>/output`. Completed `output.json` is sent to the request's
callback URL as gzip-compressed JSON. Set
`SOURCE_INVERSION_CALLBACK_URL_OVERRIDE` to force one server-controlled URL, or
set `SOURCE_INVERSION_CALLBACK_ALLOWED_HOSTS` to a comma-separated host allow
list. At least one of these restrictions is recommended in production.

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SOURCE_INVERSION_QUEUE_SIZE` | `8` | Maximum number of waiting jobs |
| `SOURCE_INVERSION_MAX_BODY_BYTES` | `10485760` | Maximum submitted JSON size |
| `SOURCE_INVERSION_CALLBACK_TIMEOUT_SECONDS` | `30` | Callback timeout per attempt |
| `SOURCE_INVERSION_CALLBACK_RETRY_DELAYS_SECONDS` | `0,60,300` | Callback retry delays |
| `SOURCE_INVERSION_MAKE_PLOTS` | `0` | Generate optional plots and GIFs |
| `SOURCE_INVERSION_RANDOM_SEED` | `0` | Algorithm random seed |
| `SOURCE_INVERSION_TEST_EPOCHS` | unset | Test-only training override |

Never set `SOURCE_INVERSION_TEST_EPOCHS` for production results. Keep one API
worker because the service owns an in-process persistent queue and runs one CPU
inversion at a time.

## CPU Docker Image

Build and run on an x86-64 Linux host with Docker:

```bash
docker build -f deployment/Dockerfile -t source-inversion:cpu .
docker compose -f deployment/docker-compose.yml up -d
```

Before deployment, replace both example tokens and set the real callback host
in `deployment/docker-compose.yml`. Job data is persisted in the
`source-inversion-data` volume under `/data/jobs`. The image uses Python 3.11,
CPU-only PyTorch, a non-root user, and one Uvicorn worker.
