# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
from logging.config import fileConfig
from pathlib import Path

import dotenv
from alembic import context
from sqlalchemy import engine_from_config, pool

from gfmstudio.config import get_settings

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = {**dotenv.dotenv_values(os.path.join(BASE_DIR, ".env")), **os.environ}

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

settings = get_settings()

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from gfmstudio.common.db import Base  # noqa: E402
from gfmstudio.fine_tuning.models import *  # noqa: F401 E402 F403

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
from gfmstudio.inference.v2.models import *  # noqa: E402 F403 F401

target_metadata = Base.metadata


def get_db_url():
    return str(settings.DATABASE_URI)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_db_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
