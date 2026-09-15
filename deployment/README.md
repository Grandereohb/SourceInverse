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
python -m deployment.entrypoint serve --host 0.0.0.0 --port 8000
```

The same entry point also exposes the isolated `worker` command used by the
service. Source development uses the existing Python-module fallback. A future
compiled image sets `SOURCE_INVERSION_EXECUTABLE` to the compiled launcher, and
the API invokes `<executable> worker ...` without requiring `python -m`.

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

## Protected Linux Image

The protected build compiles the first-party Python entry point and algorithm
modules with Nuitka in a builder stage. The runtime stage receives only the
standalone compiled distribution, runtime system libraries, and the Python base
runtime; repository source, tests, Git metadata, and compiler output are not
copied into the delivered image.

On the trusted Ubuntu x86-64 build server, first load and tag the offline base
image if it is not already present:

```bash
sha256sum -c python_3.11_slim_linux_amd64.tar.sha256
sudo docker load -i python_3.11_slim_linux_amd64.tar
sudo docker tag public.ecr.aws/docker/library/python:3.11-slim python:3.11-slim
sudo docker run --rm python:3.11-slim python --version
```

Then build and scan the protected image from the repository root:

```bash
chmod +x deployment/build_protected_image.sh deployment/scan_runtime_image.sh
sudo -E deployment/build_protected_image.sh
```

The build uses `--pull=false` and therefore requires `python:3.11-slim` to be
present in the server's local image store. Python packages and Debian build
packages are still downloaded during the build, so the trusted build server
must retain access to PyPI, the PyTorch CPU wheel index, and Debian repositories.

Create a private runtime environment file without committing its secrets, then
start the already-built image without exposing the repository as a volume:

```bash
cp deployment/runtime.env.example deployment/runtime.env
chmod 600 deployment/runtime.env
sudo docker compose \
  --env-file deployment/runtime.env \
  -f deployment/docker-compose.runtime.yml \
  up -d
```

The runtime Compose file has no `build:` section and sets `pull_policy: never`,
so the client host only needs the exported final image. Before delivery, export
and checksum that image on the trusted build server:

```bash
sudo docker save source-inversion:protected -o source-inversion_protected_amd64.tar
sha256sum source-inversion_protected_amd64.tar \
  > source-inversion_protected_amd64.tar.sha256
```

Nuitka compilation raises the cost of recovering implementation details but is
not absolute protection against a privileged host administrator. Keep the
builder stage and repository on a trusted server, and deliver only the final
runtime image archive, checksum, runtime Compose file, and environment template.
