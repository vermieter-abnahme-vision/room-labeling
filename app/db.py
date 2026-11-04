"""Engine/session setup. SQLite runs in WAL mode with short, serialised write transactions."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

IMMEDIATE_OPTION = "sqlite_immediate"
BUSY_TIMEOUT_MS = 10_000


class Database:
    """Holds the engine plus two session factories.

    * ``read_session`` – deferred transactions (readers never block under WAL).
    * ``write_session`` – ``BEGIN IMMEDIATE``: takes the write lock up-front, so
      read-modify-write sequences (submit label -> recompute consensus) are serialised
      and never fail with SQLITE_BUSY_SNAPSHOT upgrades.
    """

    def __init__(self, url: str) -> None:
        self._ensure_parent_dir(url)
        self.engine: Engine = create_engine(url, connect_args={"timeout": BUSY_TIMEOUT_MS / 1000})
        self._install_sqlite_hooks(self.engine)
        self._read_factory = sessionmaker(self.engine, expire_on_commit=False)
        self._write_factory = sessionmaker(
            self.engine.execution_options(**{IMMEDIATE_OPTION: True}), expire_on_commit=False
        )

    @staticmethod
    def _ensure_parent_dir(url: str) -> None:
        database = make_url(url).database
        if database and database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _install_sqlite_hooks(engine: Engine) -> None:
        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn: sqlite3.Connection, _record: Any) -> None:
            # Take over transaction control from pysqlite so BEGIN IMMEDIATE works.
            dbapi_conn.isolation_level = None
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            cursor.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn: Any) -> None:
            if conn.get_execution_options().get(IMMEDIATE_OPTION):
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                conn.exec_driver_sql("BEGIN")

    @contextmanager
    def read_session(self) -> Iterator[Session]:
        with self._read_factory() as session:
            yield session

    @contextmanager
    def write_session(self) -> Iterator[Session]:
        """Short write transaction: commit on success, roll back on error."""
        with self._write_factory() as session:
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def dispose(self) -> None:
        self.engine.dispose()


_database: Database | None = None


def get_database(settings: Settings | None = None) -> Database:
    global _database
    if _database is None:
        _database = Database((settings or get_settings()).database_url)
    return _database


def reset_database() -> None:
    """Dispose the process-wide database (tests)."""
    global _database
    if _database is not None:
        _database.dispose()
    _database = None
