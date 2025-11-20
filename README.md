![Geospatial Studio banner](./docs/img/banner.png)

# 🌍 GEOStudio Core

<table>
<tr>
  <td><strong>License</strong></td>
  <td>
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" />
  </td>
</tr>
<tr>
  <td><strong>TerraStackAI</strong></td>
  <td>
    <img src="https://img.shields.io/badge/TerraTorch-a3b18a" />
    <img src="https://img.shields.io/badge/TerraKit-588157" />
    <img src="https://img.shields.io/badge/Iterate-3a5a40" />
  </td>
</tr>
<tr>
  <td><strong>Built With</strong></td>
  <td>
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/FastAPI-0ba360?logo=fastapi&logoColor=white" />
    <img src="https://img.shields.io/badge/uv-4B8BBE?logo=astral&logoColor=white" />
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" />
    <img src=https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white />

  </td>
</tr>
<tr>
  <td><strong>Deployment</strong></td>
  <td>
    <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
    <img src="https://img.shields.io/badge/Podman-892ca0?logo=podman&logoColor=white" />
    <img src="https://img.shields.io/badge/Kubernetes-326ce5?logo=kubernetes&logoColor=white" />
  </td>
</tr>
</table>

[![Studio Documentation](https://img.shields.io/badge/Studio_Documentation-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://terrastackai.github.io/geospatial-studio)

## 🚀 Overview

The Geospatial Exploration and Orchestration Studio (GEOStudio) is an integrated platform for fine-tuning, inference, and orchestration of geospatial AI models. It combines a no-code UI, low-code SDK, and APIs to make working with geospatial data and AI accessible to everyone, from researchers to developers.

GEOStudio Core provides the unified API platform that orchestrates the entire geospatial machine learning workflow — from dataset onboarding and model training to model deployment for inferencing. It's codebase provides API endpoints for three core components:

- **Inference APIs** for running ML model predictions on models trained on geospatial imagery.
- **Dataset Factory APIs** for onboarding and managing labeled training datasets.
- **Fine-Tuning APIs** for orchestrating Kubernetes-based model training jobs with experiment tracking.

For details on deployment of the GEOStudio to a local or managed K8 cluster, [see here](https://github.com/terrastackai/geospatial-studio).

## 🏗 Architecture

![Workflow](docs/img/architecture-v2.drawio.png)

---

## 🏁 Getting Started

### Prerequisites

Before running GEOStudio Core locally or inside a containerized environment, ensure that all required backend services are installed, configured, and reachable. GEOStudio Core depends on both a relational database and an in-memory datastore for full functionality.

The following services must be available **before the application can start**:

<table>
<thead>
  <tr>
    <th><strong>Service</strong></th>
    <th><strong>Supported Version(s)</strong></th>
    <th><strong>Purpose</strong></th>
    <th><strong>Installation / Setup</strong></th>
  </tr>
</thead>
<tbody>
  <!-- Python -->
  <tr>
    <td align="center">
      <img src="https://www.python.org/static/community_logos/python-logo.png" width="90"><br/>
      <strong>Python</strong>
    </td>
    <td>3.11</td>
    <td>
      Required runtime for GEOStudio Core application, tooling, and CLI scripts.<br/>
    </td>
    <td>
      <a href="https://www.python.org/downloads/">Python Installation Guide</a><br/>
      <em>Recommended:</em> Use <strong>pyenv</strong> for version management.
    </td>
  </tr>

  <!-- PostgreSQL -->
  <tr>
    <td align="center">
      <img src="https://www.postgresql.org/media/img/about/press/elephant.png" width="80"><br/>
      <strong>PostgreSQL</strong>
    </td>
    <td> 15.x </td>
    <td>
      Primary datastore for GEOStudio core metadata.
    </td>
    <td>
      <a href="https://www.postgresql.org/download/">PostgreSQL Installation Guide</a><br/>
      <em>Alternatively:</em> Use a managed PostgreSQL service such as <strong>IBM Cloud Databases for PostgreSQL</strong>.
    </td>
  </tr>

  <!-- Redis -->
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/redis/redis-io/master/public/images/redis-logo.png" width="80"><br/>
      <strong>Redis</strong>
    </td>
    <td>8.x</td>
    <td>
      In-memory datastore used for:
      <ul>
        <li>Caching</li>
        <li>Message queuing for async/long-running tasks</li>
      </ul>
    </td>
    <td>
      <a href="https://redis.io/docs/latest/operate/oss_and_stack/install/">Redis Installation Guide</a><br/>
      <em>Alternatively:</em> Use a managed Redis instance.
    </td>
  </tr>

</tbody>
</table>

#### Service Configuration Requirements

To ensure the application starts successfully:

1. **PostgreSQL must be running** and accessible (local instance or cloud-hosted).  
2. Your `.env` file must contain a valid `DATABASE_URL` in the following format:

    ```sh
    DATABASE_URL=postgresql+psycopg://username:password@hostname:5432/database_name
    ```

3. If running Redis locally, ensure the default port `6379` is reachable, or update `REDIS_URL` in `.env`.

### [🖥️ Run Locally (with uv)](#local-setup-with-venv)

This project uses [uv](https://docs.astral.sh/uv/) for environment management, dependency resolution, and running commands. 
Follow the installation instructions from the [official uv docs](https://docs.astral.sh/uv/getting-started/installation/).

After installing uv, you can proceed with these GEOStudio core setup instructions 👇

```bash
# Clone the repo
git clone git@github.com:terrastackai/geospatial-studio-core.git
cd geospatial-studio-core

# Install dependencies
uv sync
source .venv/bin/activate

# Make a copy of the environment variables file and replace values where necessary
cp .env.example .env

# Run migrations to setup the database
alembic upgrade head && alembic -n auth upgrade head

# Add seed data to your database.
# TODO: Convert to a task
python scripts/load_data.py
# Create user and api key
python scripts/create_user.py

# Start the server
uv run geostudio-core dev
```

> Visit: http://localhost:3300/docs

> [!IMPORTANT]
> Developers: Make sure to enable pre-commit hooks. See the full workflow in [CONTRIBUTING.md](CONTRIBUTING.md).


### 🦭 Run with Podman

![Podman](https://img.shields.io/badge/Containerized%20with-Podman-892ca0?logo=podman&logoColor=white)

Make a copy of the environment variables file for docker/podman and replace values where necessary

```bash
cp .env.docker.example .env.docker
```

Build and run the app using [Podman](https://podman.io/):

```bash
podman build --platform linux/amd64 -t us.icr.io/gfmaas/gfmstudio-gateway --load .
podman run -p 3300:8080 --env-file .env.docker us.icr.io/gfmaas/gfmstudio-gateway:latest
```

> Visit: http://localhost:3300/docs

### 🐳 Run with Docker

![Docker](https://img.shields.io/badge/Containerized%20with-Docker-2496ED?logo=docker&logoColor=white)

Make a copy of the environment variables file for docker/podman and replace values where necessary

```bash
cp .env.docker.example .env.docker
```

Build and run the app using [Docker](https://docker.io/):

```bash
docker build --platform linux/amd64 -t us.icr.io/gfmaas/gfmstudio-gateway --load .
docker run -p 3300:8080  --env-file .env.docker us.icr.io/gfmaas/gfmstudio-gateway:latest
```

> Visit: http://localhost:3300/docs

---

## 📁 Project Structure

```bash
geospatial-model-gateway/
├── gfmstudio/
│   ├── main.py              # Entry point for the app
│   ├── auth/                # Authentication definitions
│   ├── db_migrations/       # Alembic Database files
│   │   ├── seed_data/       # Seed data to setup your intial db data.   
│   │   ├── versions/        # Alemic database schema changes
│   │   └── env.py           # Alembic config file to include sqlaclhemy models
│   ├── fine_tuning/         # FineTuning and DatasetFactory APIs
│   │   ├── core/
│   │   ├── scripts/
│   │   ├── templates/
│   │   ├── models.py        # Database models/table definition in SqlAlchemy
│   │   ├── schemas.py       # Endpoint payload/response schema definitions
│   │   └── api.py           # Endpoints/Route definitions
│   ├── inference/           # Inference APIs
│   │   ├── v2/
│   │   │   ├── api.py       # Endpoints/Route definitions
│   │   │   ├── models.py    # Database models/table definition in SqlAlchemy
│   │   │   ├── schemas.py   # Endpoint payload/response schema definitions
│   │   │   └── services.py
│   │   └── v1/
│   ├── Jira/                # JIRA Issue Endpoints
├── tests/                   # Unit and integration tests
│   ├── integration/
│   └── unit/
├── tools/                   # Supporting tools and services
│   ├── geoserver/           # Geoserver dockerfile
│   └── mlflow/              # Dockerfile to sync mlflow and studio
│   ├── terratorch/          # Terratorch dockerfiles
│   │   ├── caikit-base/     # Terratorch dockerfile for caikit (used for permanently deployed inference)
│   │   ├── pytorch-base/    # Terratorch dockerfile for pytorch (used for run inference pipeline)
│   │   ├── ubi-base/        # Terratorch dockerfile for ubi (used for finetuning job)
│   │   ├── vllm-base/       # Terratorch dockerfile (TBA)
├── Dockerfile
├── docker-compose.yml       # For local dev
├── tiltfile                 # Tilt dev config
├── alembic.ini              # Alembic DB config
├── requirements.txt
└── README.md

```

---

## 🧪 Testing & Coverage

GEOStudio Core includes a full testing toolchain based on **pytest**, **coverage**, and **static analysis tooling**.  
These tests are executed locally and in CI/CD pipelines to maintain code quality, stability, and reliability.

### 🔬 Running the Test Suite

To execute all unit tests:

```bash
pytest tests/
```

### 🧹 Static Analysis & Code Quality

This project uses [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistent formatting before commits are made.

Run these commands to setup pre-commit hooks. The pre-commit hooks will automatically lint, format and check your changes for commited credentials.

  ```bash
  pip install -r requirements-dev.txt
  pre-commit install
  ```

You could also Lint your changes with:

```bash
ruff check gfmstudio/ tests/
```

and to format with black:

```bash
# Format code
black gfmstudio/ tests/
```

---

## 🤝 Contributing

We welcome contributions from the community! See the **[CONTRIBUTING](CONTRIBUTING.md)** docs for full guidelines.
