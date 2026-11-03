#!/usr/bin/env python3
"""LLM Atlas v2 · 课程制索引构建器

Source of truth:
  课程/<领域文件夹>/*.md        每日课程笔记（frontmatter + note_md）
  课程/串讲/*.md                串讲复习笔记
  动态/*.md                      每周动态简报
  config/agents.json            课程路径（learning_path）

Derived (全部可重建):
  state/coverage.json           课程进度
  knowledge/<id>.md             各领域知识树 MOC
  课程地图.md / 术语库.md / 首页.md / 动态索引.md
  dashboard/index.html          网页看板（内嵌数据，离线可用）

Usage: python3 scripts/index.py
"""
from __future__ import annotations

import json
import re
import yaml
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "agents.json"
COURSE = ROOT / "课程"
REVIEWS = COURSE / "串讲"
NEWS = ROOT / "动态"
TOPICS = ROOT / "专题"
STATE = ROOT / "state"
KNOWLEDGE = ROOT / "knowledge"
DASH = ROOT / "dashboard"
RUNS = ROOT / "runs"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    header = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    meta: dict = {}
    try:
        loaded = yaml.safe_load(header)
        if isinstance(loaded, dict):
            meta = loaded
    except Exception:
        meta = {}
    if not meta:
        # 兼容旧式逐行解析（YAML 解析失败时兜底）
        for line in header.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value[:1] in "[{":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            meta[key.strip()] = value
    return meta, body

def parse_term_table(body: str) -> list[dict]:
    """解析笔记中「术语卡」表格（| 术语 | 解释 |）。"""
    terms: list[dict] = []
    in_section = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = "术语" in stripped
            continue
        if not in_section or not stripped.startswith("|"):
            continue
        if re.match(r"^\|[\s:|-]+\|", stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] == "术语" and cells[1] == "解释":
            continue
        if len(cells) >= 2 and cells[0] and cells[1]:
            terms.append({"term": cells[0], "explanation": cells[1]})
    return terms


def collect_urls(text: str) -> set[str]:
    return set(re.findall(r"https?://[^\s)\]>\"']+", text))


def read_lesson(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if meta.get("type") in {"supplement", "topic", "topic-index"}:
        return None
    date = str(meta.get("date") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None
    review_md = ""
    rp = RUNS / date / f"review-{path.stem}.md"
    if not rp.exists():
        rp = RUNS / date / "review.md"
    if rp.exists():
        review_md = rp.read_text(encoding="utf-8")
    urls = collect_urls(body)
    return {
        "date": date,
        "title": str(meta.get("title") or body.splitlines()[0].lstrip("# ").strip()),
        "domain": str(meta.get("domain") or ""),
        "module_id": str(meta.get("module_id") or ""),
        "module": str(meta.get("module") or ""),
        "status": str(meta.get("status") or "已学"),
        "lesson_no": meta.get("lesson_no"),
        "goals": meta.get("goals") or [],
        "tags": meta.get("tags") or [],
        "markdown": body.strip(),
        "review_md": review_md,
        "term_count": len(parse_term_table(body)),
        "source_count": len(urls),
        "terms": parse_term_table(body),
        "path": str(path.relative_to(ROOT)),
    }


def read_review(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    date = str(meta.get("date") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None
    return {
        "date": date,
        "title": str(meta.get("title") or body.splitlines()[0].lstrip("# ").strip()),
        "round": meta.get("round"),
        "tags": meta.get("tags") or [],
        "markdown": body.strip(),
    }


def read_news(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    date = str(meta.get("date") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None
    return {
        "date": date,
        "title": str(meta.get("title") or body.splitlines()[0].lstrip("# ").strip()),
        "week": str(meta.get("week") or ""),
        "tags": meta.get("tags") or [],
        "markdown": body.strip(),
    }


def read_topic(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if meta.get("type") != "topic":
        return None
    date = str(meta.get("date") or "")
    urls = collect_urls(body)
    return {
        "date": date,
        "title": str(meta.get("title") or body.splitlines()[0].lstrip("# ").strip()),
        "topic": str(meta.get("topic") or ""),
        "topic_id": str(meta.get("topic_id") or ""),
        "part": str(meta.get("part") or ""),
        "part_title": str(meta.get("part_title") or ""),
        "order": meta.get("order"),
        "tags": meta.get("tags") or [],
        "markdown": body.strip(),
        "term_count": len(parse_term_table(body)),
        "source_count": len(urls),
        "terms": parse_term_table(body),
        "path": str(path.relative_to(ROOT)),
    }


def read_topic_index(path: Path) -> dict | None:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    if meta.get("type") != "topic-index":
        return None
    return {
        "topic_id": str(meta.get("topic_id") or path.stem),
        "title": str(meta.get("title") or body.splitlines()[0].lstrip("# ").strip()),
        "date": str(meta.get("date") or ""),
        "path": str(path.relative_to(ROOT)),
    }


def build_progress(config: dict, lessons: list[dict]) -> dict:
    path = config["learning_path"]
    modules: dict = {}
    for m in path:
        modules[m["id"]] = {
            "status": "planned",
            "first_date": "",
            "last_date": "",
            "lesson_files": [],
        }
    for lesson in sorted(lessons, key=lambda x: x["date"]):
        mid = lesson["module_id"]
        if mid not in modules:
            continue
        entry = modules[mid]
        if entry["status"] == "planned":
            entry["first_date"] = lesson["date"]
        entry["status"] = lesson["status"]
        entry["last_date"] = lesson["date"]
        entry["lesson_files"].append(lesson["path"])

    current = None
    for m in path:
        if modules[m["id"]]["status"] != "已学":
            current = m
            break

    recent = [l["path"] for l in lessons[-4:]]
    last = recent[-1] if recent else ""
    review_rounds = 0
    for f in sorted(REVIEWS.glob("*.md")) if REVIEWS.exists() else []:
        try:
            meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
            review_rounds = max(review_rounds, int(meta.get("round") or 0))
        except Exception:
            pass

    domain_ids = {d["folder"]: d["id"] for d in config["domains"]}
    domains: dict = {
        d["id"]: {"learned_modules": [], "first_seen": {}, "tags": [], "last_date": ""}
        for d in config["domains"]
    }
    for lesson in sorted(lessons, key=lambda x: x["date"]):
        did = domain_ids.get(lesson["domain"])
        if not did:
            continue
        entry = domains[did]
        entry["last_date"] = lesson["date"]
        if lesson["module_id"] and lesson["module_id"] not in entry["learned_modules"]:
            entry["learned_modules"].append(lesson["module_id"])
        entry["first_seen"].setdefault(lesson["module_id"], lesson["date"])
        entry["tags"] = sorted(set(entry["tags"]) | set(lesson["tags"]))

    return {
        "mode": "course",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lesson_count": len(lessons),
        "review_count": len([f for f in REVIEWS.glob("*.md")]) if REVIEWS.exists() else 0,
        "news_count": len([f for f in NEWS.glob("*.md")]) if NEWS.exists() else 0,
        "current_module": current["id"] if current else None,
        "current_module_name": current["name"] if current else None,
        "last_lesson_path": last,
        "recent_lesson_paths": recent,
        "review_rounds": review_rounds,
        "modules": modules,
        "domains": domains,
    }


def write_state(progress: dict) -> None:
    STATE.mkdir(exist_ok=True)
    (STATE / "coverage.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_mocs(config: dict, progress: dict, lessons: list[dict]) -> None:
    KNOWLEDGE.mkdir(exist_ok=True)
    path_by_id = {m["id"]: m for m in config["learning_path"]}
    domain_ids = {d["folder"]: d["id"] for d in config["domains"]}
    for domain in config["domains"]:
        did = domain["id"]
        entry = progress["domains"][did]
        learned = set(entry["learned_modules"])
        first_seen = entry["first_seen"]
        mods = [m for m in config["learning_path"] if m["domain"] == did]
        lines = [
            "---",
            f"aliases: [{domain['name']}]",
            f"tags: [domain/{did}]",
            "---",
            "",
            f"# {domain['emoji']} {domain['name']}",
            "",
            f"> {domain['mission']}",
            "",
            f"**进度**：{len(learned)}/{len(mods)} 个模块 · 最近学习 {entry['last_date'] or '—'} · 累计标签 {len(entry['tags'])} 个",
            "",
            "## 课程模块",
            "",
            "| 模块 | 状态 | 首次学习 | 笔记 |",
            "| --- | --- | --- | --- |",
        ]
        for m in mods:
            done = m["id"] in learned
            status = "✅ 已学" if done else "⬜ 待学"
            date = first_seen.get(m["id"], "—")
            links = "、".join(
                f"[[{Path(p).stem}|{Path(p).stem[:10]}]]"
                for p in progress["modules"][m["id"]]["lesson_files"]
            )
            lines.append(f"| {m['name']} | {status} | {date} | {links or '—'} |")
        records = [l for l in lessons if domain_ids.get(l["domain"]) == did]
        lines += ["", "## 学习记录", ""]
        if records:
            lines += ["| 日期 | 课题 | 模块 |", "| --- | --- | --- |"]
            lines += [
                f"| [[{Path(r['path']).stem}|{r['date']}]] | {r['title']} | {r['module']} |"
                for r in sorted(records, key=lambda x: x["date"])
            ]
        else:
            lines.append("_还没有课程笔记。_")
        others = [d for d in config["domains"] if d["id"] != did]
        lines += [
            "",
            "## 关联",
            "",
            "回到 [[首页]] · [[课程地图]] · [[术语库]]",
            "其他领域：" + " · ".join(f"[[{o['id']}|{o['emoji']} {o['name']}]]" for o in others),
            "",
        ]
        (KNOWLEDGE / f"{did}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_course_map(config: dict, progress: dict, lessons: list[dict] | None = None) -> None:
    domain_by_id = {d["id"]: d for d in config["domains"]}
    total_days = sum(m["days"] for m in config["learning_path"])
    done_days = sum(m["days"] for m in config["learning_path"] if progress["modules"][m["id"]]["status"] == "已学")
    lines = [
        "# 🗺️ 课程地图",
        "",
        "> 学习路径：**推理优先、昇腾 + vLLM 主栈**，共 " + str(len(config["learning_path"])) + " 个模块 / 约 " + str(total_days) + " 天。进度：已学 " + str(done_days) + " 天。",
        "",
        f"> 当前模块：**{progress['current_module_name'] or '已完成全部课程'}** · 已上 {progress['lesson_count']} 课 · 串讲 {progress['review_count']} 次",
        "",
        "## 学习路径",
        "",
        "| # | 模块 | 领域 | 预计天数 | 状态 | 笔记 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ln_by_path = {l["path"]: l.get("lesson_no") for l in (lessons or [])}
    for i, m in enumerate(config["learning_path"], 1):
        st = progress["modules"][m["id"]]
        status = {"planned": "⬜ 待学", "进行中": "🔄 进行中", "已学": "✅ 已学"}.get(st["status"], st["status"])
        d = domain_by_id[m["domain"]]
        files = sorted(st["lesson_files"], key=lambda p: ln_by_path.get(p) or 0)
        links = "、".join(
            f"[[{Path(p).stem}|第{ln_by_path.get(p) or i}课]]"
            for p in files
        )
        lines.append(f"| {i} | {m['name']} | {d['emoji']} {d['name']} | {m['days']} | {status} | {links or '—'} |")
    lines += [
        "",
        "## 领域入口",
        "",
        " · ".join(f"[[{d['id']}|{d['emoji']} {d['name']}]]" for d in config["domains"]),
        " · [[术语库]] · [[动态索引]] · [[首页]]",
        "",
    ]
    (ROOT / "课程地图.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_terms(config: dict, lessons: list[dict]) -> None:
    domain_ids = {d["folder"]: d["id"] for d in config["domains"]}
    domain_by_id = {d["id"]: d for d in config["domains"]}
    groups: dict[str, list] = {d["id"]: [] for d in config["domains"]}
    seen: set = set()
    for lesson in sorted(lessons, key=lambda x: x["date"]):
        did = domain_ids.get(lesson["domain"])
        if not did:
            continue
        for t in lesson["terms"]:
            key = (did, t["term"])
            if key in seen:
                continue
            seen.add(key)
            groups[did].append({**t, "lesson": lesson})
    total = sum(len(v) for v in groups.values())
    lines = [
        "# 📇 术语库",
        "",
        f"> 全部课程术语的自动聚合（{total} 个），按领域分组。每天读一遍新术语，比背单词有用。",
        "",
        "回到 [[首页]] · [[课程地图]]",
        "",
    ]
    for did, terms in groups.items():
        if not terms:
            continue
        d = domain_by_id[did]
        lines += [f"## {d['emoji']} {d['name']}", ""]
        lines += ["| 术语 | 解释 | 出自 |", "| --- | --- | --- |"]
        for t in terms:
            lesson_link = f"[[{Path(t['lesson']['path']).stem}|{t['lesson']['date']}]]"
            expl = t["explanation"].replace("|", "\\|")
            lines.append(f"| {t['term']} | {expl} | {lesson_link} |")
        lines.append("")
    if total == 0:
        lines.append("_还没有术语，上完第一课就会出现。_")
    (ROOT / "术语库.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_home(config: dict, progress: dict, lessons: list[dict], news: list[dict], topics: list[dict] | None = None) -> None:
    domain_by_id = {d["id"]: d for d in config["domains"]}
    latest = lessons[-1] if lessons else None
    total_mods = len(config["learning_path"])
    learned_mods = sum(1 for m in config["learning_path"] if progress["modules"][m["id"]]["status"] == "已学")
    lines = [
        "---",
        "aliases: [首页, Home]",
        "tags: [atlas/home]",
        "---",
        "",
        "# 🗺️ LLM Atlas · 大模型学习系统",
        "",
        "> 一个每天自动运转的 AI 学习小组：总策划、讲师、助教、主编，按课程路径每天一课，帮一个基础较弱的工程师把大模型地基打牢。**推理优先，昇腾 + vLLM 主栈。**",
        "",
        "## 今天",
        "",
        f"- 📖 当前课程：**{progress['current_module_name'] or '全部学完'}**" + (f"（最近一课：[[{Path(latest['path']).stem}|{latest['title']}]]）" if latest else ""),
        "- 🗺️ [[课程地图]] · [[术语库]] · [[动态索引]]",
        "- 📊 网页看板：`dashboard/index.html`（浏览器打开）",
        "",
        "## 领域",
        "",
        " · ".join(f"[[{d['id']}|{d['emoji']} {d['name']}]]" for d in config["domains"]),
        "",
        "## 进度",
        "",
        f"已上 **{progress['lesson_count']}** 课 · 模块 **{learned_mods}/{total_mods}** · 串讲 {progress['review_count']} 次 · 术语 {sum(len(l['terms']) for l in lessons)} 个",
        "",
        "## 最近课程",
        "",
    ]
    for l in sorted(lessons, key=lambda x: (x["date"], int(x["lesson_no"]) if x["lesson_no"] else 0))[-6:][::-1]:
        lines.append(f"- [[{Path(l['path']).stem}|{l['date']}]] {l['title']}")
    if news:
        lines += ["", "## 最近动态", ""]
        for n in news[-3:][::-1]:
            lines.append(f"- [[动态/{n['week']}|{n['title']}]]")
    if topics:
        by_id: dict = {}
        for t in topics:
            by_id.setdefault(t["topic_id"], t)
        lines += ["", "## 专题", ""]
        for tid, t in by_id.items():
            folder = t["path"].split("/")[1]
            lines.append(f"- [[专题/{folder}/{folder}|{t['topic']}（专题）]]")
    lines += [
        "",
        "## 使用提示",
        "",
        "- 📅 每天 08:30 自动生成新一课（需要 DSH 会话在线；错过可手动说「补跑今天的课」）",
        "- ✍️ 想改课程顺序/重点：编辑 `config/agents.json` 的 learning_path",
        "- 🧭 想看图：打开 [[知识地图.canvas|知识地图 Canvas]]",
        "- 🗂️ 本页由 `scripts/index.py` 生成，请勿手改",
        "",
    ]
    (ROOT / "首页.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_news_index(news: list[dict]) -> None:
    lines = [
        "# 📰 动态简报索引",
        "",
        "> 每周一份的行业动态，与课程完全分开。碎片时间看，正文课程不受影响。",
        "",
        "## 全部简报",
        "",
        "| 周 | 标题 |",
        "| --- | --- |",
    ]
    for n in sorted(news, key=lambda x: x["week"], reverse=True):
        lines.append(f"| {n['week']} | [[动态/{n['week']}|{n['title']}]] |")
    lines += ["", "回到 [[首页]]", ""]
    (ROOT / "动态索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_topic_index(topics: list[dict], indexes: list[dict]) -> None:
    by_id: dict[str, list[dict]] = {}
    for t in topics:
        by_id.setdefault(t["topic_id"], []).append(t)
    lines = [
        "# 🧪 专题索引",
        "",
        "> 课程路径之外的独立知识专题：按一个主题，从模型演进 / 推理框架 / 硬件三个视角分别讲透。想深入哪个主题都可以随时加。",
        "",
    ]
    for t in sorted(topics, key=lambda x: (x["topic_id"], x["order"] or 0)):
        by_id[t["topic_id"]] = by_id.get(t["topic_id"], [])
    for tid, group in by_id.items():
        name = group[0]["topic"]
        latest = max(t["date"] for t in group)
        lines += [f"## {name}", "", f"更新于 {latest} · [[专题/{group[0]['path'].split('/')[1]}/{group[0]['path'].split('/')[1]}|总览]]", ""]
        lines += ["| 视角 | 笔记 |", "| --- | --- |"]
        for t in sorted(group, key=lambda x: x["order"] or 0):
            lines.append(f"| {t['part_title']} | [[专题/{t['path'].split('/')[1]}/{t['part']}|{t['title']}]] |")
        lines.append("")
    if not by_id:
        lines.append("_还没有专题笔记。想开一个专题直接说，例如「帮我整理 Mamba 家族的知识」。_")
    lines += ["回到 [[首页]]", ""]
    (ROOT / "专题索引.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_canvas() -> None:
    p = ROOT / "知识地图.canvas"
    if p.exists():
        return
    nodes = [
        {"id": "home", "type": "file", "file": "首页.md", "x": 0, "y": 0, "width": 300, "height": 140},
        {"id": "map", "type": "file", "file": "课程地图.md", "x": 0, "y": -560, "width": 320, "height": 150},
        {"id": "terms", "type": "file", "file": "术语库.md", "x": -700, "y": -560, "width": 280, "height": 130},
        {"id": "news", "type": "file", "file": "动态索引.md", "x": 700, "y": -560, "width": 280, "height": 130},
        {"id": "fd", "type": "file", "file": "knowledge/foundation.md", "x": -1080, "y": -240, "width": 300, "height": 160},
        {"id": "me", "type": "file", "file": "knowledge/model-evolution.md", "x": -1080, "y": 120, "width": 300, "height": 160},
        {"id": "if", "type": "file", "file": "knowledge/inference-frameworks.md", "x": 1080, "y": -240, "width": 300, "height": 160},
        {"id": "hw", "type": "file", "file": "knowledge/hardware.md", "x": 1080, "y": 120, "width": 300, "height": 160},
        {"id": "tr", "type": "file", "file": "knowledge/training.md", "x": 360, "y": 360, "width": 300, "height": 160},
        {"id": "note", "type": "text", "text": "每天 08:30 自动生成一课\n网页看板：dashboard/index.html\n课程配置：config/agents.json", "x": -360, "y": 360, "width": 320, "height": 100},
    ]
    edges = [
        {"id": "e1", "fromNode": "home", "toNode": "map", "fromSide": "top", "toSide": "bottom"},
        {"id": "e2", "fromNode": "home", "toNode": "fd", "fromSide": "left", "toSide": "right"},
        {"id": "e3", "fromNode": "home", "toNode": "me", "fromSide": "left", "toSide": "right"},
        {"id": "e4", "fromNode": "home", "toNode": "if", "fromSide": "right", "toSide": "left"},
        {"id": "e5", "fromNode": "home", "toNode": "hw", "fromSide": "right", "toSide": "left"},
        {"id": "e6", "fromNode": "home", "toNode": "tr", "fromSide": "bottom", "toSide": "top"},
        {"id": "e7", "fromNode": "home", "toNode": "terms", "fromSide": "left", "toSide": "right"},
        {"id": "e8", "fromNode": "home", "toNode": "news", "fromSide": "right", "toSide": "left"},
        {"id": "e9", "fromNode": "me", "toNode": "fd", "fromSide": "bottom", "toSide": "top"},
        {"id": "e10", "fromNode": "if", "toNode": "hw", "fromSide": "bottom", "toSide": "top"},
        {"id": "e11", "fromNode": "if", "toNode": "tr", "fromSide": "right", "toSide": "left"},
        {"id": "e12", "fromNode": "hw", "toNode": "tr", "fromSide": "bottom", "toSide": "top"},
    ]
    p.write_text(json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dashboard(config: dict, progress: dict, lessons: list[dict], reviews: list[dict], news: list[dict], topics: list[dict] | None = None) -> None:
    domain_by_folder = {d["folder"]: d for d in config["domains"]}
    domain_curriculum: dict[str, list] = {d["id"]: [] for d in config["domains"]}
    for m in config["learning_path"]:
        st = progress["modules"][m["id"]]
        domain_curriculum[m["domain"]].append(
            {"id": m["id"], "name": m["name"], "level": m["level"], "status": st["status"], "date": st["first_date"]}
        )
    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": "course",
        "domains": [
            {
                "id": d["id"],
                "folder": d["folder"],
                "name": d["name"],
                "emoji": d["emoji"],
                "color": d["color"],
                "mission": d["mission"],
                "curriculum": domain_curriculum[d["id"]],
            }
            for d in config["domains"]
        ],
        "stats": {
            "lessons": len(lessons),
            "modules": sum(1 for m in config["learning_path"] if progress["modules"][m["id"]]["status"] == "已学"),
            "terms": len({(l["domain"], t["term"]) for l in lessons for t in l["terms"]}),
        },
        "coverage": {did: {"learned_modules": progress["domains"][did]["learned_modules"], "first_seen": progress["domains"][did]["first_seen"]} for did in domain_curriculum},
        "days": [
            {
                "id": Path(l["path"]).stem,
                "date": l["date"],
                "title": l["title"],
                "main_thread": " · ".join(l["goals"][:3]) if l["goals"] else "",
                "domain": l["domain"],
                "module": l["module"],
                "module_id": l["module_id"],
                "lesson_no": l["lesson_no"],
                "tags": l["tags"],
                "markdown": l["markdown"],
                "review_md": l["review_md"],
                "term_count": l["term_count"],
                "source_count": l["source_count"],
                "briefs": {},
                "coverage": {},
                "generated": l["date"],
            }
            for l in sorted(lessons, key=lambda x: (x["date"], int(x["lesson_no"]) if x["lesson_no"] else 0), reverse=True)
        ],
        "reviews": [{"date": r["date"], "round": r["round"], "title": r["title"], "tags": r["tags"], "markdown": r["markdown"]} for r in sorted(reviews, key=lambda x: x["date"], reverse=True)],
        "news": [{"week": n["week"], "date": n["date"], "title": n["title"], "tags": n["tags"], "markdown": n["markdown"]} for n in sorted(news, key=lambda x: x["week"], reverse=True)],
        "topics": [
            {
                "date": t["date"], "title": t["title"], "topic": t["topic"], "topic_id": t["topic_id"],
                "part": t["part"], "part_title": t["part_title"], "order": t["order"], "tags": t["tags"],
                "markdown": t["markdown"], "term_count": t["term_count"], "source_count": t["source_count"],
            }
            for t in sorted(topics or [], key=lambda x: (x["topic_id"], x["order"] or 0))
        ],
    }
    template = (DASH / "template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (DASH / "index.html").write_text(template.replace("/*__DATA__*/", blob), encoding="utf-8")


def main() -> None:
    config = load_config()
    lessons, reviews, news = [], [], []
    if COURSE.exists():
        for folder in sorted(COURSE.iterdir()):
            if not folder.is_dir() or folder.name == "串讲":
                continue
            for f in sorted(folder.glob("*.md")):
                l = read_lesson(f)
                if l:
                    lessons.append(l)
        if REVIEWS.exists():
            for f in sorted(REVIEWS.glob("*.md")):
                r = read_review(f)
                if r:
                    reviews.append(r)
    if NEWS.exists():
        for f in sorted(NEWS.glob("*.md")):
            n = read_news(f)
            if n:
                news.append(n)
    topics, topic_indexes = [], []
    if TOPICS.exists():
        for folder in sorted(TOPICS.iterdir()):
            if not folder.is_dir():
                continue
            for f in sorted(folder.glob("*.md")):
                t = read_topic(f)
                if t:
                    topics.append(t)
                    continue
                ti = read_topic_index(f)
                if ti:
                    topic_indexes.append(ti)
    lessons.sort(key=lambda x: x["date"])

    progress = build_progress(config, lessons)
    write_state(progress)
    write_mocs(config, progress, lessons)
    write_course_map(config, progress, lessons)
    write_terms(config, lessons)
    write_home(config, progress, lessons, news, topics)
    write_news_index(news)
    write_topic_index(topics, topic_indexes)
    write_canvas()
    build_dashboard(config, progress, lessons, reviews, news, topics)

    learned = sum(1 for m in config["learning_path"] if progress["modules"][m["id"]]["status"] == "已学")
    print(f"indexed {len(lessons)} lessons, {len(reviews)} reviews, {len(news)} news, {len(topics)} topic notes")
    print(f"learned modules: {learned}/{len(config['learning_path'])} | current: {progress['current_module_name']}")
    print(f"dashboard -> {DASH / 'index.html'}")


if __name__ == "__main__":
    main()
