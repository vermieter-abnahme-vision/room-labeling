"""Import a labelled subset of MIT Indoor67 or Places365 into the room scheme.

Images land in the normal image store (source=library, group_id=library:<dataset>:<class>) and
receive a label from the pseudo-labeler "library". They feed probe training but are excluded from
agreement statistics. Scene classes without a mapping below are skipped.

Datasets
  indoor67   download manually (http only, 2.4 GB): https://web.mit.edu/torralba/www/indoor.html
             extract indoorCVPR_09.tar and pass the folder that contains ``Images/``.
  places365  either pass an extracted dir containing val_256/, places365_val.txt and
             categories_places365.txt, or use --download (official HTTPS URLs, validation set).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.labelschemes import ROOM_SCHEME
from app.services import ingest
from scripts._common import bootstrap

logger = logging.getLogger("scripts.import_library")

# --- Mapping tables: dataset scene class -> room label (unlisted classes are skipped) -------------
INDOOR67_MAP: dict[str, str] = {
    "bathroom": "bad",
    "bedroom": "schlafzimmer",
    "children_room": "schlafzimmer",
    "kitchen": "kueche",
    "livingroom": "wohnzimmer",
    "corridor": "flur",
}
PLACES365_MAP: dict[str, str] = {
    "/b/bathroom": "bad",
    "/s/shower": "bad",
    "/b/bedroom": "schlafzimmer",
    "/b/bedchamber": "schlafzimmer",
    "/c/childs_room": "schlafzimmer",
    "/k/kitchen": "kueche",
    "/l/living_room": "wohnzimmer",
    "/t/television_room": "wohnzimmer",
    "/c/corridor": "flur",
    "/e/entrance_hall": "flur",
}

# Fixed official locations (no user-supplied URLs are ever fetched -> no SSRF).
PLACES365_FILELIST_URL = (
    "https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar"
)
PLACES365_VAL_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
MAX_DOWNLOAD_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class LibraryItem:
    path: Path
    source_class: str
    label: str


def _subset(
    items: dict[str, list[Path]], mapping: dict[str, str], max_per_class: int, seed: int
) -> Iterator[LibraryItem]:
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not cryptography
    missing = sorted(set(mapping) - set(items))
    if missing:
        logger.warning("mapped classes not found in dataset: %s", ", ".join(missing))
    for source_class in sorted(mapping):
        files = sorted(items.get(source_class, []))
        rng.shuffle(files)
        for path in files[:max_per_class]:
            yield LibraryItem(path, source_class, mapping[source_class])


def scan_indoor67(
    root: Path, mapping: dict[str, str], max_per_class: int, seed: int
) -> list[LibraryItem]:
    base = root / "Images" if (root / "Images").is_dir() else root
    items: dict[str, list[Path]] = defaultdict(list)
    for cls in mapping:
        folder = base / cls
        if folder.is_dir():
            items[cls] = [p for p in folder.iterdir() if p.is_file() and not p.is_symlink()]
    return list(_subset(items, mapping, max_per_class, seed))


def _read_categories(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, idx = line.rsplit(" ", 1)
        out[int(idx)] = name.strip()
    return out


def scan_places365(
    root: Path, mapping: dict[str, str], max_per_class: int, seed: int
) -> list[LibraryItem]:
    items: dict[str, list[Path]] = defaultdict(list)
    val_list, categories = root / "places365_val.txt", root / "categories_places365.txt"
    if val_list.is_file() and categories.is_file():
        names = _read_categories(categories)
        val_dir = root / "val_256"
        for line in val_list.read_text(encoding="utf-8").splitlines():
            fname, idx = line.split()
            cls = names.get(int(idx), "")
            path = val_dir / fname
            if cls in mapping and path.is_file() and not path.is_symlink():
                items[cls].append(path)
    elif (root / "data_256").is_dir():  # train layout: data_256/<letter>/<class>/*.jpg
        for cls in mapping:
            folder = root / "data_256" / cls.lstrip("/")
            if folder.is_dir():
                items[cls] = [p for p in folder.iterdir() if p.is_file() and not p.is_symlink()]
    else:
        raise SystemExit(f"{root}: no Places365 layout found (see --help)")
    return list(_subset(items, mapping, max_per_class, seed))


# --- Download (Places365 validation set, HTTPS only) ----------------------------------------
def _download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    if not url.startswith("https://"):
        raise ValueError("only https downloads are allowed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": "room-labeling-import"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as resp, dest.open("wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("download exceeds size limit")
            digest.update(chunk)
            out.write(chunk)
    logger.info("downloaded %s (%d bytes) sha256=%s", dest.name, size, digest.hexdigest())
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name}")


def _extract_members(tar_path: Path, wanted: dict[str, Path]) -> None:
    """Extract only the wanted members (no path from the archive is ever used for writing)."""
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            target = wanted.get(Path(member.name).name)
            if target is None:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


def download_places365_val(root: Path, mapping: dict[str, str], sha256: str | None) -> None:
    filelist = root / "filelist.tar"
    _download(PLACES365_FILELIST_URL, filelist)
    wanted_lists = {name: root / name for name in ("places365_val.txt", "categories_places365.txt")}
    _extract_members(filelist, wanted_lists)
    names = _read_categories(wanted_lists["categories_places365.txt"])
    keep = {
        line.split()[0]
        for line in wanted_lists["places365_val.txt"].read_text(encoding="utf-8").splitlines()
        if names.get(int(line.split()[1]), "") in mapping
    }
    val_tar = root / "val_256.tar"
    _download(PLACES365_VAL_URL, val_tar, sha256)
    _extract_members(val_tar, {name: root / "val_256" / name for name in keep})
    filelist.unlink(missing_ok=True)
    val_tar.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", choices=["indoor67", "places365"], required=True)
    parser.add_argument("--root", type=Path, required=True, help="extracted dataset folder")
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--download", action="store_true", help="places365 only: fetch val set into --root"
    )
    parser.add_argument("--sha256", help="optional expected sha256 of val_256.tar")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    mapping = INDOOR67_MAP if args.dataset == "indoor67" else PLACES365_MAP
    scheme = ctx.registry.get(ROOM_SCHEME)
    skipped_labels = sorted({v for v in mapping.values() if v not in scheme.labels})
    if skipped_labels:
        logger.warning("labels not in ROOM_LABELS, classes skipped: %s", skipped_labels)
        mapping = {k: v for k, v in mapping.items() if v in scheme.labels}

    if args.download:
        if args.dataset != "places365":
            parser.error("--download is only supported for places365")
        download_places365_val(args.root, mapping, args.sha256)

    scan = scan_indoor67 if args.dataset == "indoor67" else scan_places365
    items = scan(args.root, mapping, args.max_per_class, args.seed)
    logger.info(
        "%d candidate images in %d mapped classes", len(items), len({i.source_class for i in items})
    )

    report = ingest.IngestReport()
    with ctx.db.write_session() as session:
        for n, item in enumerate(items, 1):
            try:
                _, created = ingest.ingest_library_image(
                    session,
                    ctx.settings,
                    item.path,
                    dataset=args.dataset,
                    source_class=item.source_class.strip("/").replace("/", "_"),
                    label=item.label,
                    registry=ctx.registry,
                )
            except ingest.IngestError as exc:
                report.skipped += 1
                report.errors.append(f"{item.path.name}: {exc}")
                continue
            report.added += created
            report.duplicates += not created
            if n % 200 == 0:
                session.commit()  # keep write transactions short
    logger.info(
        "library import: %d added, %d duplicates, %d skipped",
        report.added,
        report.duplicates,
        report.skipped,
    )
    for err in report.errors[:20]:
        logger.warning("skipped: %s", err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
