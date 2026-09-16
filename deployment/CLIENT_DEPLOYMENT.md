# SourceInverse Client Deployment

The client host receives only the protected Linux amd64 Docker image and
runtime configuration. It does not require the SourceInverse repository,
Python packages, Nuitka, or build tools.

## 1. Verify and load

Run these commands inside the delivered package directory:

```bash
sha256sum -c source-inversion-*.docker.tar.gz.sha256
sudo docker load -i source-inversion-*.docker.tar.gz
sudo docker image inspect source-inversion:protected \
  --format '{{.Os}}/{{.Architecture}}'
```

The expected platform is `linux/amd64`.

## 2. Configure

```bash
cp runtime.env.example runtime.env
chmod 600 runtime.env
```

Edit `runtime.env` and replace `SOURCE_INVERSION_API_TOKEN`. Configure either
`SOURCE_INVERSION_CALLBACK_URL_OVERRIDE` or
`SOURCE_INVERSION_CALLBACK_ALLOWED_HOSTS` for the client's callback endpoint.
Do not set `SOURCE_INVERSION_TEST_EPOCHS` in production.

## 3. Start and verify

```bash
sudo docker compose --env-file runtime.env \
  -f docker-compose.runtime.yml up -d
curl -fsS http://127.0.0.1:8000/api/v1/health
sudo docker compose --env-file runtime.env \
  -f docker-compose.runtime.yml ps
```

The health response should be `{"status":"ok"}`.

## 4. Operations

View logs:

```bash
sudo docker compose --env-file runtime.env \
  -f docker-compose.runtime.yml logs -f --tail 200
```

Stop the service without deleting job data:

```bash
sudo docker compose --env-file runtime.env \
  -f docker-compose.runtime.yml down
```

Job data is stored in the Docker volume `source-inversion-data`. Do not use
`docker compose down -v` unless permanent deletion of all jobs is intended.
