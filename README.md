


## Dockerized API and batch runtimes

Project 4 provides two local Docker application images.

Implemented images:

| Image | Purpose |
|---|---|
| `project4-anomaly-api:local` | Online FastAPI anomaly inference runtime |
| `project4-anomaly-batch:local` | Batch/offline anomaly scoring runtime |

The API container runs FastAPI internally on port `8004`. Docker Compose publishes it to host port `8014` by default so it does not conflict with local `uvicorn` runs on `8004`.

Implemented Docker scope:

- API image build using `docker/Dockerfile.api`.
- Batch image build using `docker/Dockerfile.batch`.
- API runtime on `http://localhost:8014`.
- Batch runtime exposed through the `batch` Compose profile.
- Active production model `v002` packaged into both images from `artifacts/models`.
- Runtime configuration packaged from `configs`.
- Prediction and alert logs mounted from the host through `./logs:/app/logs`.
- Batch scoring input mounted from `./data/features/batch_scoring:/app/data/features/batch_scoring`.
- Prometheus metrics exposed by the API container.
- Docker healthcheck against `/health`.

Build both images directly:

    docker build -f docker/Dockerfile.api -t project4-anomaly-api:local .
    docker build -f docker/Dockerfile.batch -t project4-anomaly-batch:local .

Run the API with Docker Compose:

    docker compose up -d --build anomaly-api

Validate the API runtime:

    docker compose ps
    curl http://localhost:8014/health
    curl http://localhost:8014/model/active
    curl http://localhost:8014/metrics | head -50

Run the batch runtime help contract:

    docker compose --profile batch run --rm anomaly-batch

Run batch scoring with a mounted feature file:

    docker compose --profile batch run --rm anomaly-batch \
      python -m anomaly_detection.batch_inference \
      --input-path /app/data/features/batch_scoring/features.jsonl \
      --output-path /app/logs/predictions/batch_predictions.jsonl \
      --entity-type customer \
      --source-project project_4 \
      --source-table local_batch_scoring

Use a different API host port if needed:

    PROJECT4_API_HOST_PORT=8020 docker compose up -d --build anomaly-api
    curl http://localhost:8020/health

Stop the Docker runtime:

    docker compose down --remove-orphans

Current Docker scope:

- Implemented: local API container build.
- Implemented: local batch container build.
- Implemented: Docker Compose API runtime.
- Implemented: Docker Compose batch profile.
- Implemented: model `v002` loads inside the containerized API.
- Implemented: Prometheus metrics are exposed from the containerized API.
- Planned: CI Docker build validation.
- Optional/planned: Kubernetes manifests.
- Not implemented: cloud deployment.

## CI/CD validation

GitHub Actions workflow:

- `.github/workflows/ci.yml`
- Workflow name: `ML Platform Validation`

Implemented CI gates:

- Ruff code quality validation.
- Basic secret pattern scan.
- SQL smoke checks for required schema/table/query files.
- Training snapshot and Isolation Forest training tests.
- Real-source extraction smoke tests.
- Model registry tests.
- Baseline metrics audit for active production model `v002`.
- Drift, alert, and rollback tests.
- FastAPI, metrics, and prediction logging tests.
- Full pytest suite.
- Docker API image build using `docker/Dockerfile.api`.
- Docker batch image build using `docker/Dockerfile.batch`.

Completion status:

- Implemented locally: workflow file and validation commands.
- Fully complete only after GitHub Actions passes on the remote branch or pull request.
- Not implemented: cloud deployment.

