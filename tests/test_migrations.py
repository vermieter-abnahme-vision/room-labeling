from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import text

from alembic import command
from app.db import Database
from app.models import Base


def test_migrations_match_models(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    cfg = Config("alembic.ini")
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")

    db = Database(url)
    try:
        with db.engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diff == []
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    finally:
        db.dispose()
    command.downgrade(cfg, "base")


def test_write_sessions_serialise_with_immediate_lock(tmp_path: Path) -> None:
    """Two concurrent writers must not fail with 'database is locked' (busy_timeout + IMMEDIATE)."""
    import threading

    from sqlalchemy import select

    from app.models import Labeler

    db = Database(f"sqlite:///{tmp_path / 'c.db'}")
    Base.metadata.create_all(db.engine)
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                with db.write_session() as s:
                    s.add(Labeler(oidc_sub=f"w{n}-{i}", display_name="x"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with db.read_session() as s:
        assert len(list(s.scalars(select(Labeler)))) == 80
    db.dispose()
