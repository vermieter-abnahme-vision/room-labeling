"""Generate SYNTHETIC demo data (coloured rooms with text) for local end-to-end tests.

Writes ./images_in/demo_<n>/*.jpg and, if ffmpeg exists, ./videos_in/demo_video.mp4.
No real photos are involved; both folders are git-ignored.
"""

from __future__ import annotations

import argparse
import colorsys
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOMS = ["Bad", "Kueche", "Wohnzimmer", "Schlafzimmer", "Flur"]


def make_image(path: Path, room: str, index: int) -> None:
    hue = (ROOMS.index(room) / len(ROOMS) + index * 0.013) % 1.0
    r, g, b = (int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.45, 0.85))
    img = Image.new("RGB", (960, 640), (r, g, b))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 920, 600), outline=(255, 255, 255), width=6)
    draw.text((70, 70), f"DEMO {room} #{index}", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=85)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apartments", type=int, default=6)
    parser.add_argument("--per-room", type=int, default=3)
    parser.add_argument("--images-in", type=Path, default=Path("images_in"))
    parser.add_argument("--videos-in", type=Path, default=Path("videos_in"))
    args = parser.parse_args(argv)

    count = 0
    for apt in range(1, args.apartments + 1):
        for room in ROOMS:
            for i in range(args.per_room):
                make_image(
                    args.images_in / f"demo_{apt}" / f"{room.lower()}_{i}.jpg", room, apt * 10 + i
                )
                count += 1
    print(f"wrote {count} synthetic images to {args.images_in}/")

    if shutil.which("ffmpeg"):
        args.videos_in.mkdir(parents=True, exist_ok=True)
        target = args.videos_in / "demo_video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x480:rate=10",
                "-t",
                "8",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ],
            check=True,
        )
        print(f"wrote {target}")
    else:
        print("ffmpeg not found - skipped demo video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
