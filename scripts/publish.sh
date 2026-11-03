#!/usr/bin/env bash
# 一键发布：重建索引 → 健康检查 → 构建站点 → 提交并推送
set -euo pipefail
cd "$(dirname "$0")/.."

# 从 .secrets/github_token 加载 token（该目录已 gitignore）
if [ -z "${GH_TOKEN:-}" ] && [ -f .secrets/github_token ]; then
  export GH_TOKEN="$(cat .secrets/github_token)"
fi
export GH_TOKEN

python3 scripts/index.py
python3 scripts/healthcheck.py
python3 scripts/build_site.py

if git remote get-url origin >/dev/null 2>&1; then
  git add -A
  git commit -m "update: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M')" || true
  git push
  echo "✅ 已推送。若开了 GitHub Pages / Cloudflare Pages，1 分钟内线上更新。"
else
  cat <<'EOT'
还没有配置远程仓库，二选一：
  A. GitHub（公开仓库，免费 Pages）：
     gh auth login
     gh repo create llm-atlas --public --source=. --push
  B. 已有仓库：
     git remote add origin <你的仓库地址>
     git push -u origin main
EOT
fi
