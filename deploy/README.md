# 部署到手机随时访问

> ✅ **已部署**：https://linnea-lin-00638949.github.io/llm-atlas/
> 仓库：https://github.com/linnea-lin-00638949/llm-atlas（public）
> 每日课程更新后执行 `bash scripts/publish.sh` 即可同步线上。


看板是一个**单文件静态站点**（构建产物在 `_site/`），任何静态托管都能用。四种方案按「上手成本 → 随时可用」排序：

---

## 方案 A · 同 Wi-Fi 局域网（0 成本，现在就能用）

```bash
python3 scripts/serve.py --open
```

终端会打印两个地址：

```
电脑访问：http://127.0.0.1:8765/index.html
手机访问：http://10.131.203.16:8765/index.html   （需与 Mac 同一 Wi-Fi）
```

手机浏览器打开「手机访问」那一行即可。只暴露看板，不暴露笔记源文件。

**限制**：Mac 必须开着、同一 Wi-Fi。适合临时看，不算「随时」。

---

## 方案 B · GitHub Pages（真正随时访问，推荐）

一次性配置：

```bash
gh auth login                                  # 浏览器授权一次
gh repo create llm-atlas --public --source=. --push
```

然后到仓库 **Settings → Pages → Build and deployment → Source 选 “GitHub Actions”**。
工作流 `.github/workflows/pages.yml` 已就绪，push 后自动构建部署。

之后每次更新（新课程、新日报）：

```bash
bash scripts/publish.sh
```

访问地址：`https://<你的用户名>.github.io/llm-atlas/`

**注意**：免费 Pages 要求仓库 public，意味着学习笔记公开可读（页面已加 `noindex`，但知道地址就能看）。介意的话用方案 C。

---

## 方案 C · Cloudflare Pages（仓库可私有）

1. 在 GitHub 建**私有**仓库并推送本目录；
2. Cloudflare Dashboard → Workers & Pages → Create → Pages → Connect to Git，选该仓库；
3. 构建设置：
   - Build command：`python scripts/build_site.py`
   - Build output directory：`_site`
4. 保存后每次 push 自动部署，地址形如 `https://llm-atlas.pages.dev`；
5. 想限制访问：在 Cloudflare Zero Trust → Access 给该项目加一条策略（邮箱验证）。

---

## 方案 D · 手机上直接看笔记（Obsidian）

看板只是浏览入口，**课程正文在 Obsidian 库**里。把 `Learning` 文件夹放进 iCloud Drive / 坚果云，手机安装 Obsidian → 打开该库，即可随时看课程、术语库、课程地图，双链与图谱都能用。

适合「读」；看板适合「快速浏览 + 分享」。

---

## 加到主屏幕（像 App 一样打开）

- **iOS Safari**：分享 → 添加到主屏幕
- **Android Chrome**：菜单 → 添加到主屏幕

已配置 `manifest.webmanifest` 与图标（192/512），添加后会显示名为「LLM Atlas」的图标，全屏无地址栏打开。

---

## 每次更新后

```bash
bash scripts/publish.sh
```

它依次执行：重建索引 → 健康检查 → 构建 `_site/` → 提交并推送（若配置了远程仓库）。

---

## 说明

- 看板数据内嵌在 HTML 里，当前 **450 KB**；课程到 50 课以上时建议改成按日期懒加载（`dashboard/data/*.json`），届时我可以改造。
- 页面已加 `noindex`，但静态托管本身不提供鉴权；要私密请用方案 C + Cloudflare Access。
