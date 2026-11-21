from alembic import context

from ops_agent.database import create_database_engine
from ops_agent.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connection = config.attributes.get("connection")
    if connection is not None:
        run_migrations(connection)
    else:
        engine = create_database_engine()
        try:
            with engine.connect() as connection:
                run_migrations(connection)
        finally:
            engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported.")
else:
    run_migrations_online()
