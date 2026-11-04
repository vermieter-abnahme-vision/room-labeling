from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.labelschemes import SchemeRegistry
from app.models import Annotation, Image
from app.services import ingest
from scripts import import_library as lib
from scripts.import_library import (
    INDOOR67_MAP,
    PLACES365_MAP,
    scan_indoor67,
    scan_places365,
)
from tests.conftest import write_jpeg


def test_mapping_targets_exist_in_default_label_set(registry: SchemeRegistry) -> None:
    labels = set(registry.get("room").labels)
    assert set(INDOOR67_MAP.values()) <= labels
    assert set(PLACES365_MAP.values()) <= labels


def test_scan_indoor67_skips_unmapped_classes(tmp_path: Path) -> None:
    for cls, n in (("kitchen", 5), ("bathroom", 2), ("casino", 4)):
        for i in range(n):
            write_jpeg(tmp_path / "Images" / cls / f"{i}.jpg", (i * 20, 3, 4))
    items = scan_indoor67(tmp_path, INDOOR67_MAP, max_per_class=3, seed=1)
    assert {i.source_class for i in items} == {"kitchen", "bathroom"}  # casino unmapped
    assert sum(i.source_class == "kitchen" for i in items) == 3  # capped
    assert {i.label for i in items} == {"kueche", "bad"}
    assert items == scan_indoor67(tmp_path, INDOOR67_MAP, max_per_class=3, seed=1)  # reproducible


def test_scan_places365_val_layout_and_import(
    tmp_path: Path, session: Session, settings: Settings, registry: SchemeRegistry
) -> None:
    (tmp_path / "categories_places365.txt").write_text(
        "/b/bathroom 45\n/k/kitchen 203\n/a/airfield 0\n"
    )
    names = {"v1.jpg": 203, "v2.jpg": 45, "v3.jpg": 0}
    (tmp_path / "places365_val.txt").write_text("".join(f"{n} {i}\n" for n, i in names.items()))
    for i, n in enumerate(names):
        write_jpeg(tmp_path / "val_256" / n, (i * 50, 9, 9))
    items = scan_places365(tmp_path, PLACES365_MAP, 10, 0)
    assert {(i.source_class, i.label) for i in items} == {
        ("/k/kitchen", "kueche"),
        ("/b/bathroom", "bad"),
    }
    for item in items:
        ingest.ingest_library_image(
            session,
            settings,
            item.path,
            dataset="places365",
            source_class=item.source_class.strip("/").replace("/", "_"),
            label=item.label,
            registry=registry,
        )
    groups = {g for (g,) in session.execute(select(Image.group_id))}
    assert groups == {"library:places365:k_kitchen", "library:places365:b_bathroom"}
    assert session.scalar(select(func.count()).select_from(Annotation)) == 2


def test_download_rejects_non_https(tmp_path: Path) -> None:

    with pytest.raises(ValueError, match="https"):
        lib._download("http://example.test/x.tar", tmp_path / "x.tar")


def test_scripts_refuse_unmigrated_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from app.config import get_settings
    from scripts._common import bootstrap

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="make migrate"):
            bootstrap()
        from alembic.config import Config

        from alembic import command

        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")
        bootstrap().db.dispose()
    finally:
        get_settings.cache_clear()
