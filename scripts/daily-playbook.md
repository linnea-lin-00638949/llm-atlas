# 每日运行手册（课程制 · 给编排 Agent）

> 目标：每天产出一课 `课程/<领域>/<日期>-<模块id>.md`，并让 Obsidian 笔记与看板自动更新。
> 时区 Asia/Shanghai，日期一律 `YYYY-MM-DD`。

---

## 0. 读取上下文

```bash
cd /Users/lyf02488571/Desktop/Learning
cat state/coverage.json | head -30
```

关键字段：

- `lesson_count`：已上课程数（判断是否到串讲日：`lesson_count > 0 且 % 4 == 0`）；
- `current_module` / `current_module_name`：当前该学的模块；
- `last_lesson_path` / `recent_lesson_paths`：最近 1/4 课的相对路径。

---

## 1. 组装 workflow 参数

脚本：`scripts/workflow-course.js`（读全文作为 workflow 的 script）。

```json
{
  "date": "YYYY-MM-DD",
  "mode": "auto",
  "config_path": "/Users/lyf02488571/Desktop/Learning/config/agents.json",
  "root": "/Users/lyf02488571/Desktop/Learning",
  "coverage": { "lesson_count": N, "current_module": "...", "modules": { ... } },
  "prev_lesson": "<last_lesson_path 或空>",
  "last_lessons": [
    { "path": "课程/基础/2026-11-03-nn-basics-1.md", "title": "..." }
  ]
}
```

- `coverage` 直接取 `state/coverage.json` 的主体（modules 太大时可只保留 lesson_count/current_module/review_count，策划会读 config 自己排）；
- `mode`：`auto`（默认）按规则自动：每 4 课 → `review`；周一 → `news`；否则 `lesson`。手动指定可强制；
- meta：

```json
{
  "name": "llm-atlas-course",
  "description": "课程制学习工作流：每日一课（排课→备课→助教→主编→归档）",
  "phases": [
    {"title": "01 · 排课"}, {"title": "02 · 备课"}, {"title": "03 · 助教审校"},
    {"title": "04 · 成稿"}, {"title": "05 · 归档"}, {"title": "完成"}
  ]
}
```

建议 `run_in_background: true`。

---

## 2. 返回结构

正常时（归档 Agent 已落盘并重建索引）：

```json
{
  "date": "...", "mode": "lesson",
  "plan": { "module_id": "...", "day_title": "...", "status": "已学" },
  "title": "...",
  "review_verdict": "pass|revise",
  "review_issue_count": 0,
  "archive": { "ok": true, "files": ["课程/基础/2026-11-03-nn-basics-1.md", "..."], "index_output": "..." }
}
```

若 `archive.ok` 为 false，返回体带 `note` / `plan_full` / `review` 与 `note_file`，手工落盘（frontmatter 模板见 `templates/课程模板.md`），再跑 `python3 scripts/index.py`。

---

## 3. 校验（幂等，可重复跑）

```bash
python3 scripts/index.py
ls 课程/<领域>/
cat state/coverage.json | head -20
```

确认：

- 新笔记存在，frontmatter 完整（date/title/domain/module_id/module/status/lesson_no/goals/tags）；
- `state/coverage.json` 的 lesson_count 增加、current_module 推进；
- Obsidian 打开后：课程地图状态更新、术语库新增术语、图谱新增节点；
- 看板 `dashboard/index.html`：新课出现在左侧、术语/来源计数正确。

## 3.5 发布到手机看板

```bash
bash scripts/publish.sh
```

重建索引 → 健康检查 → 构建 `_site/` → 提交推送；GitHub Actions 自动部署到
<https://linnea-lin-00638949.github.io/llm-atlas/>（约 1 分钟）。token 从 `.secrets/github_token` 读取（已 gitignore）。
没网或没 token 时跳过即可，不影响本地。

---

## 4. 异常处理

| 情况 | 处理 |
| --- | --- |
| 讲师/主编失败 | workflow 内自动重试 2 次；仍失败就补跑当天（不推进课程，进度由笔记存在性决定） |
| 归档失败 | 按返回体手工落盘 + `python3 scripts/index.py` |
| 某课助教给出 high 问题 | 主编已在成稿时修复；若问题仍在，下一轮串讲里补充修正，并把结论记入 `runs/<date>/review.md` |
| 上一课没学完（用户要求重学） | 把 `state/coverage.json` 里该模块 status 改回 `planned`，`python3 scripts/index.py` 重建后重新排课 |
| 用户要求跳过某模块 | 同上，把该模块 status 改为 `已学`（first_date 留空即可） |
| 网络不可用 | 课程正文只依赖稳定来源，讲师可凭教材/文档写，但「动态简报」停止运行并明确标注未联网 |

---

## 5. 质量红线（课程制）

- 课程正文禁止以新闻为主体；新闻只进 `动态/`；
- 每个术语第一次出现必须解释；不偷用没学过的知识；
- worked example 的每个数字要能重算；助教负责核对；
- 不确定写「存疑」；不编造公式、数字、URL；
- 每课必须有自测题 + 详细答案（`<details>` 折叠）。
