#!/usr/bin/env python3
"""把看板打包成可部署的静态站点（_site/）。"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
SITE = ROOT / "_site"


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    for name in ["index.html", "manifest.webmanifest", "icon.svg", "icon-192.png", "icon-512.png"]:
        src = DASH / name
        if src.exists():
            shutil.copy2(src, SITE / name)
    (SITE / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")
    (SITE / ".nojekyll").touch()
    total = sum(f.stat().st_size for f in SITE.iterdir())
    print(f"site built: {SITE} | {len(list(SITE.iterdir()))} files | {total/1024:.0f} KB")


if __name__ == "__main__":
    main()
