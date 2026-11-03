#!/usr/bin/env python3
"""Vault 健康检查：课程文件、进度、生成物是否一致，防止文件被误移/误删。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config" / "agents.json").read_text())
STATE = json.loads((ROOT / "state" / "coverage.json").read_text())

warn, ok = [], []
course = ROOT / "课程"
import yaml

def note_type(p):
    try:
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return "lesson"
        head = text[3:text.find("\n---", 3)].strip()
        meta = yaml.safe_load(head) or {}
        return str(meta.get("type") or "lesson")
    except Exception:
        return "lesson"

all_files = [p for p in course.glob("*/*.md") if p.parent.name != "串讲"]
lesson_files = [p for p in all_files if note_type(p) != "supplement"]
supplement_files = [p for p in all_files if note_type(p) == "supplement"]
review_files = list((course / "串讲").glob("*.md"))
if STATE["lesson_count"] != len(lesson_files):
    warn.append(f"state.lesson_count={STATE['lesson_count']} 但课程文件={len(lesson_files)}")
else:
    ok.append(f"课程文件 {len(lesson_files)} 篇与进度一致")
ok.append(f"补充阅读 {len(supplement_files)} 篇（不计入课程）")
if STATE["review_count"] != len(review_files):
    warn.append(f"state.review_count={STATE['review_count']} 但串讲文件={len(review_files)}")
else:
    ok.append(f"串讲 {len(review_files)} 篇与进度一致")

for m in CONFIG["learning_path"]:
    st = STATE["modules"][m["id"]]
    if st["status"] == "已学" and not st["lesson_files"]:
        warn.append(f"模块 {m['id']} 标记已学但没有课程文件")
    for rel in st["lesson_files"]:
        if not (ROOT / rel).exists():
            warn.append(f"进度引用的文件不存在：{rel}")

for name in ["首页.md", "课程地图.md", "术语库.md", "dashboard/index.html", "state/coverage.json"]:
    (ok if (ROOT / name).exists() else warn).append(("存在 " if (ROOT / name).exists() else "缺失 ") + name)

print("== OK ==")
for x in ok:
    print("  ✓", x)
if warn:
    print("== WARN ==")
    for x in warn:
        print("  !", x)
    sys.exit(1)
print("全部一致。")
