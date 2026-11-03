#!/usr/bin/env python3
"""收集 疑问/ 收件箱里的提问，按课程分组打印，方便逐条讲解。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "疑问"


def main() -> None:
    files = sorted(p for p in INBOX.glob("*.md") if p.name != "README.md")
    if not files:
        print("收件箱是空的：把「我的疑问」导出的 .md 放进 疑问/ 即可。")
        return
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        print(f"\n===== {f.name} =====")
        print(text.strip())
        total += text.count("- 我的问题")
    print(f"\n共 {len(files)} 个文件 / 约 {total} 条问题")


if __name__ == "__main__":
    main()
