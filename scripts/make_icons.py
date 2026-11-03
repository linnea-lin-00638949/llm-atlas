#!/usr/bin/env python3
"""生成看板图标：深色圆角底 + 双色菱形标记。"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "dashboard"

def diamond(draw, cx, cy, r, width, fill=None, color=None):
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    if fill:
        draw.polygon(pts[:-1], fill=fill)
    if color:
        draw.line(pts, fill=color, width=width, joint="curve")

def make(size: int, path: Path):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(size * 0.04)
    r = int(size * 0.22)
    # 圆角背景
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=r, fill=(9, 13, 22, 255),
                        outline=(38, 48, 72, 255), width=max(2, size // 128))
    # 柔光点
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(24, 0, -1):
        a = int(10 * (i / 24))
        rr = int(size * 0.46 * (i / 24))
        gd.ellipse([size // 2 - rr, size // 2 - rr, size // 2 + rr, size // 2 + rr],
                   fill=(110, 140, 255, a))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)
    cx = cy = size // 2
    w = max(6, size // 24)
    # 外圈菱形（蓝）与内圈菱形（绿）
    diamond(d, cx, cy, int(size * 0.30), w, color=(138, 164, 255, 255))
    diamond(d, cx, cy, int(size * 0.17), max(4, w - size // 48), color=(89, 224, 182, 255))
    d.ellipse([cx - size * 0.026, cy - size * 0.026, cx + size * 0.026, cy + size * 0.026],
              fill=(235, 242, 255, 255))
    img.save(path)

def make_svg(path: Path):
    path.write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#8aa4ff"/><stop offset="1" stop-color="#59e0b6"/>
    </linearGradient>
  </defs>
  <rect x="20" y="20" width="472" height="472" rx="112" fill="#090d16" stroke="#263048" stroke-width="4"/>
  <path d="M256 110 L402 256 L256 402 L110 256 Z" fill="none" stroke="#8aa4ff" stroke-width="20" stroke-linejoin="round"/>
  <path d="M256 185 L327 256 L256 327 L185 256 Z" fill="none" stroke="#59e0b6" stroke-width="14" stroke-linejoin="round"/>
  <circle cx="256" cy="256" r="13" fill="#ebf2ff"/>
</svg>
''', encoding="utf-8")

if __name__ == "__main__":
    make(192, OUT / "icon-192.png")
    make(512, OUT / "icon-512.png")
    make_svg(OUT / "icon.svg")
    print("icons written:", (OUT / "icon-192.png").stat().st_size, (OUT / "icon-512.png").stat().st_size)
