#!/bin/sh
# Apply DB migrations (idempotent), then run the given command.
set -eu
alembic upgrade head
exec "$@"
