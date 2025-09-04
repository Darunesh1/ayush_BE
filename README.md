## AyushSync Backend

AyushSync Backend is a Django-powered API service containerized with Docker and backed by PostgreSQL for a smooth, reproducible local development workflow. It runs with an ASGI server (Uvicorn) for rapid reloads, applies database migrations on startup, and uses Compose Watch to sync code changes into the running container for fast iteration.

### Key features
- Django + ASGI (Uvicorn) with auto-reload for development.
- PostgreSQL 16 with a named volume for persistent local data.
- One-command setup using Docker Compose (build and up).
- Simple commands for migrations, creating a superuser, and accessing the admin.


### Clone the repository (HTTPS)
```bash
git clone https://github.com/your-org/ayushsync-backend.git
cd ayushsync-backend
```

### Or clone with SSH (requires SSH keys configured)
```bash
git clone git@github.com:your-org/ayushsync-backend.git
cd ayushsync-backend
```

### Optionally, clone a specific branch
```bash
git clone -b main https://github.com/your-org/ayushsync-backend.git
cd ayushsync-backend
```
### Copy the example env and fill details

```bash
cp env_example .env
```

## To run the application

```bash
docker compose build

docker compose up
```

### Open an interactive shell using the Compose service name (recommended)
```bash
docker exec -it ayushsync_backend sh
```

### Create super user in interactive shell
```bash
uv run manage.py createsuperuser
```
now give the user name password

now u can access the admin dashboard

if any error

run in interactive shell
```bash
uv run manage.py collectstatic
```