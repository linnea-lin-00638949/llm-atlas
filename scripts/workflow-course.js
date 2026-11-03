// LLM Atlas v2 · 课程制学习工作流
// 模式由编排层或日程规则决定：
//   lesson  — 每日一课（讲师备课 → 助教审校 → 主编成稿 → 归档）
//   review  — 每 4 课一次串讲复习（顺带出本周动态简报）
//   news    — 每周动态简报（单独文件，与课程分开）
// args: { date, config_path, root, coverage, mode, prev_lesson, last_lessons }

const A = args || {};
const date = A.date || new Date().toISOString().slice(0, 10);
const configPath = A.config_path || '/Users/lyf02488571/Desktop/Learning/config/agents.json';
const root = A.root || '/Users/lyf02488571/Desktop/Learning';
const coverage = A.coverage || {};
const prevLesson = A.prev_lesson || '';
const lastLessons = A.last_lessons || [];
const lessonCount = coverage.lesson_count || 0;

const dow = new Date(date + 'T12:00:00').getDay(); // 0=周日 1=周一
let mode = A.mode;
if (!mode || mode === 'auto') {
  if (lessonCount > 0 && lessonCount % 4 === 0) mode = 'review';
  else if (dow === 1) mode = 'news';
  else mode = 'lesson';
}
const doNews = mode === 'review' || mode === 'news' || dow === 1;

function isoWeek(d) {
  const date = new Date(d + 'T12:00:00Z');
  const dayNum = (date.getDay() + 6) % 7;
  date.setDate(date.getDate() - dayNum + 3);
  const firstThursday = new Date(date.getFullYear(), 0, 4);
  const fdn = (firstThursday.getDay() + 6) % 7;
  firstThursday.setDate(firstThursday.getDate() - fdn + 3);
  const week = String(1 + Math.round((date - firstThursday) / 604800000)).padStart(2, '0');
  return date.getFullYear() + '-W' + week;
}

const planSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['module_id', 'module_name', 'domain_id', 'folder', 'day_title', 'goals', 'pre_test', 'focus', 'connect_prev', 'status', 'lesson_no', 'next_up'],
  properties: {
    module_id: { type: 'string' },
    module_name: { type: 'string' },
    domain_id: { type: 'string' },
    folder: { type: 'string' },
    day_title: { type: 'string' },
    goals: { type: 'array', items: { type: 'string' } },
    pre_test: { type: 'array', items: { type: 'string' } },
    focus: { type: 'array', items: { type: 'string' } },
    connect_prev: { type: 'string' },
    status: { type: 'string', enum: ['已学', '进行中'] },
    lesson_no: { type: 'integer' },
    next_up: { type: 'string' }
  }
};

const lessonBodySchema = {
  type: 'object',
  additionalProperties: false,
  required: ['lesson_md', 'terms', 'sources'],
  properties: {
    lesson_md: { type: 'string' },
    terms: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['term', 'explanation'],
        properties: { term: { type: 'string' }, explanation: { type: 'string' } }
      }
    },
    sources: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'url', 'kind'],
        properties: { title: { type: 'string' }, url: { type: 'string' }, kind: { type: 'string' } }
      }
    }
  }
};

const reviewSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'summary', 'issues'],
  properties: {
    verdict: { type: 'string', enum: ['pass', 'revise'] },
    summary: { type: 'string' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'claim', 'problem', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          claim: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' }
        }
      }
    }
  }
};

const noteSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'note_md', 'goals', 'tags'],
  properties: {
    title: { type: 'string' },
    note_md: { type: 'string' },
    goals: { type: 'array', items: { type: 'string' } },
    tags: { type: 'array', items: { type: 'string' } }
  }
};

const newsSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'brief_md', 'tags'],
  properties: {
    title: { type: 'string' },
    brief_md: { type: 'string' },
    tags: { type: 'array', items: { type: 'string' } }
  }
};

const archiveSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['ok', 'files', 'index_output'],
  properties: {
    ok: { type: 'boolean' },
    files: { type: 'array', items: { type: 'string' } },
    index_output: { type: 'string' }
  }
};

async function runAgent(prompt, opts, attempts = 2) {
  for (let i = 0; i < attempts; i++) {
    const r = await agent(prompt, opts);
    if (r) return r;
    log(`重试 ${opts.label || 'agent'}（${i + 1}/${attempts}）`);
  }
  return null;
}

let result = { date, mode };

if (mode === 'lesson') {
  phase('01 · 排课');
  log('总策划正在按课程路径确定今天这一课');

  const plannerPrompt = `你是「LLM Atlas」课程的总策划。今天日期：${date}。

第一步：用 read 工具读取 ${configPath}。这是课程配置：learning_path 是唯一的学习顺序，schedule 是排课规则，audience 是学习者画像，principles 是教学原则。
第二步：${prevLesson ? `用 read 工具读上一课 ${prevLesson}，了解进度和上次讲到的深度。` : '本期是课程第一课，没有上一课。'}

课程进度（coverage）：
${JSON.stringify(coverage, null, 2)}

排课规则（严格按此执行）：
1. 今天要学的模块 = learning_path 中第一个 status 不是「已学」的模块（对照 coverage.modules 里的 status；没有记录视为未学）。
2. 若上一课属于同一模块、且该模块 days > 1，今天继续该模块（day_title 用「（下）」），status 填「已学」（今天是最后一天）。
3. 若该模块 days == 1，今天一次学完，status 填「已学」。
4. 若该模块 days > 1 且今天是第一天，status 填「进行中」。
5. lesson_no = coverage.lesson_count + 1。
6. next_up = learning_path 中下一个模块的 name（最后一个模块填「课程结束，进入实战复习」）。

输出要求：
- module_id / module_name / domain_id / folder 从 config 原样抄。
- day_title：给今天这课起具体标题，如「第 1 课 · 神经网络在算什么」。
- goals：3 条学习目标，写成「学完今天，你能……」。
- pre_test：2–3 道课前自测题，检验前置知识，题干短，不给答案。
- focus：3–5 个今天必须讲透的点。
- connect_prev：一句话说明与上一课的关系。
只输出 JSON。`;

  const plan = await runAgent(plannerPrompt, { schema: planSchema, label: '总策划' });
  if (!plan) throw new Error('总策划未能排课');
  result.plan = { module_id: plan.module_id, day_title: plan.day_title, status: plan.status };
  log(`今日课程：${plan.day_title}（${plan.module_id}，${plan.status}）`);

  phase('02 · 备课');
  log('讲师正在备课（稳定来源，从零解释）');

  const teacherPrompt = `你是「${plan.folder}」课程的讲师。今天 ${date}，${plan.day_title}。

学习者画像：${A.audience || '刚入门、基础较弱的工程师'}。他的目标是把地基打牢，不追新闻。

第一步：用 read 工具读取 ${configPath}，找到 id 为 ${plan.module_id} 的模块条目（concepts / goals / prereqs / sources_hint）。${prevLesson ? `第二步：读上一课 ${prevLesson}，保持术语与难度衔接。` : ''}

今天必须讲透的点：
${plan.focus.map((f, i) => `${i + 1}. ${f}`).join('\n')}

与上一课的关系：${plan.connect_prev}

备课要求：
1. 来源只用稳定材料：sources_hint 里的材料、官方文档、经典论文、公认教材。禁止把「最近两周的新闻」当正文。sources 里每条给出 title / url / kind（官方文档/论文/教材/教程）。
2. 每个术语第一次出现必须解释；不用未定义的英文缩写；复杂概念先直觉类比、再严格定义。
3. 必须有一个逐步推导的 worked example，带具体数字，读者能跟着手算。
4. 「常见误区」至少 2 个，指出错误的直觉并纠正。
5. 能落到工程实践就落（例如「你在昇腾上跑 vLLM 时……」）。
6. lesson_md 结构（Markdown，不要一级标题）：
## 课前自测
## 为什么需要它
## 核心概念
### <概念名>
（每个概念：定义 → 直觉 → 机制）
## 一步一步理解
## 常见误区
## 一句话总结
## 术语卡
| 术语 | 解释 |
| --- | --- |
## 下一步
## 来源
7. 中文，1800–3000 字；删除空话；公式写清楚；不确定的地方写「存疑」，不要编造。
8. terms 数组与术语卡表格一致（term + explanation）。

只输出 JSON。`;

  const lesson = await runAgent(teacherPrompt, { schema: lessonBodySchema, label: '讲师' });
  if (!lesson) throw new Error('讲师未能完成备课');

  phase('03 · 助教审校');
  log('助教正在审校：概念正确性 + 初学者友好度');

  const auditPrompt = `你是助教。今天 ${date}，课程「${plan.day_title}」（模块 ${plan.module_name}）。

这份讲义是给基础较弱的工程师看的。逐项检查：
1. 概念与数学是否正确（重点核对公式、数值例子）。
2. 初学者友好度：有没有没解释就用的术语/缩写？worked example 是否能跟着手算？
3. 有没有把「存疑/推测」写成事实？
4. 是否偷偷假设了还没学的知识（对照前置模块）？
5. 教学结构是否完整：课前自测、核心概念、例子、误区、自测题、术语卡、来源是否齐全？

讲义全文：
${JSON.stringify(lesson, null, 2)}

最多列 8 条问题，按严重程度排序（high = 概念错误，必须改；medium = 难懂或缺关键限定；low = 可改进）。没有问题就 verdict: pass、issues 为空。只输出 JSON。`;

  let review = await runAgent(auditPrompt, { schema: reviewSchema, label: '助教' });
  if (!review) review = { verdict: 'pass', summary: '助教未返回结果，按未审校处理。', issues: [] };
  result.review_verdict = review.verdict;
  result.review_issue_count = (review.issues || []).length;
  log(`审校结论：${review.verdict}｜${(review.issues || []).length} 条问题`);

  phase('04 · 成稿');
  log('主编正在成稿');

  const editorPrompt = `你是《LLM Atlas》课程主编。今天 ${date}，${plan.day_title}。

先修复审校指出的 high / medium 问题，无法修复的降级表述或删除。不要把「存疑」写成事实。

课程计划：
${JSON.stringify(plan, null, 2)}
讲义与审校意见：
${JSON.stringify({ lesson, review }, null, 2)}

note_md 必须严格使用以下结构（一级标题用 title）：

# ${plan.day_title}

> 学完今天，你能：
> 1. …
> 2. …
> 3. …

（然后依次是讲义的：课前自测 / 为什么需要它 / 核心概念 / 一步一步理解 / 常见误区 / 一句话总结，其中「课前自测」只给题不给答案）

## 自测题
### 1. 题目
<details><summary>查看答案</summary>答案</details>
（3–5 题，覆盖今天的核心概念，答案要详细）

## 术语卡
| 术语 | 解释 |
| --- | --- |

## 下一步
（说明下一课 ${plan.next_up || '内容'} 会接着讲什么）

## 来源
（逐条列 title — URL）

写作要求：中文，术语保留英文；1800–3000 字；信息密度高；每个概念都有直觉、机制、例子三层。tags 给 4–6 个可复用标签。只输出 JSON。`;

  const note = await runAgent(editorPrompt, { schema: noteSchema, label: '主编' });
  if (!note) throw new Error('主编未能成稿');
  result.title = note.title;

  phase('05 · 归档');
  log('归档 Agent 正在落盘并重建索引');

  const filename = `${date}-${plan.module_id}`;
  const archivePayload = {
    kind: 'lesson',
    folder: plan.folder,
    file_path: `${root}/课程/${plan.folder}/${filename}.md`,
    frontmatter: {
      date, title: note.title, domain: plan.folder, module_id: plan.module_id,
      module: plan.module_name, status: plan.status, lesson_no: plan.lesson_no,
      goals: note.goals, tags: note.tags
    },
    note_md: note.note_md,
    plan,
    review,
    audit_path: `${root}/runs/${date}/review.md`,
    plan_path: `${root}/runs/${date}/plan.json`
  };
  const archivePrompt = `你是归档 Agent。工作目录：${root}。日期：${date}。

任务（所有内容原样落盘，不要改写）：
1. 用 write 工具写课程笔记 ${archivePayload.file_path}：
   先写 frontmatter（三个短横线起止，值用 JSON 语法）：
   date / title / domain / module_id / module / status / lesson_no / goals（JSON 数组）/ tags（JSON 数组）
   然后空一行，写 note_md。
2. 写审校记录 ${archivePayload.audit_path}（目录先 mkdir -p）：含 verdict / summary / issues（claim、问题、建议）。
3. 写计划 ${archivePayload.plan_path}：plan 的 JSON 原样。
4. 用 bash 执行 cd ${root} && python3 scripts/index.py，确认输出含 indexed 与 dashboard。

数据：
${JSON.stringify(archivePayload, null, 2)}

完成后返回 {ok, files, index_output}。失败返回 {ok: false, ...} 并说明哪一步出错。`;

  const archive = await runAgent(archivePrompt, { schema: archiveSchema, label: '归档' });
  result.archive = archive;
  if (!archive || !archive.ok) {
    result.note = note;
    result.plan_full = plan;
    result.review = review;
    result.note_file = `${root}/课程/${plan.folder}/${filename}.md`;
    result.note = '归档失败，返回原始内容由编排者手动落盘';
  }
}

if (mode === 'review') {
  const round = Math.floor(lessonCount / 4);
  phase('01 · 串讲');
  log(`第 ${round} 轮串讲：把最近四课串成一张图`);

  const synthesisPrompt = `你是《LLM Atlas》串讲师。今天 ${date}，第 ${round} 轮串讲复习。

第一步：用 read 工具依次读取最近四课（按顺序读全）：
${lastLessons.map((l, i) => `${i + 1}. ${l.path}（${l.title}）`).join('\n')}
${lastLessons.length ? '' : `（没有上一课记录，请读取 ${root}/课程 下的最近笔记，用 bash ls 找）`}
第二步：用 read 工具读取 ${configPath} 了解课程目标。

你的任务不是重新讲一遍，而是把四课串成一张能记住的图。note_md 结构：

# 第 ${round} 轮串讲 · ${date}

> 本轮四课：……

## 一张图串起来
（用依赖关系、数据流或因果链把四课串起来，可用文字图）

## 模块速览表
| 课 | 一句话核心 | 关键公式/概念 | 与其他课的关系 |

## 综合自测
### 1. 题目
<details><summary>查看答案</summary>答案</details>
（6–8 题，跨课综合，答案详细）

## 易混淆对照表
| 容易混的概念 | 区别一句话 |

## 自查问题
（3 个问题，回答不上来就回看对应课程）

## 下一轮预告

写作要求：中文，1500–2500 字；面向基础弱的读者；只输出 JSON。`;

  const synthesis = await runAgent(synthesisPrompt, { schema: noteSchema, label: '串讲师' });
  if (!synthesis) throw new Error('串讲师未能完成串讲');

  phase('02 · 审校');
  const audit2 = await runAgent(
    `你是助教。审校这份串讲笔记，重点检查：跨课连接是否成立、综合自测答案是否正确、有没有把没学过的东西当作已学。最多 5 条问题，只输出 JSON。\n\n${JSON.stringify(synthesis, null, 2)}`,
    { schema: reviewSchema, label: '助教' }
  );
  const review2 = audit2 || { verdict: 'pass', summary: '', issues: [] };

  const note = synthesis; // 审校意见由主编并入
  phase('03 · 成稿');
  const note2 = await runAgent(
    `你是课程主编。把串讲笔记按审校意见修订后输出最终版（结构不变，只修错、不重写）。审校意见：${JSON.stringify(review2, null, 2)}\n串讲笔记：${JSON.stringify(note, null, 2)}`,
    { schema: noteSchema, label: '主编' }
  );
  const finalNote = note2 || note;
  result.title = finalNote.title;

  phase('04 · 归档');
  const reviewPayload = {
    kind: 'review',
    file_path: `${root}/课程/串讲/${date}-第${round}轮串讲.md`,
    frontmatter: { date, title: finalNote.title, type: 'review', round, tags: finalNote.tags },
    note_md: finalNote.note_md,
    audit_path: `${root}/runs/${date}/review.md`,
    review: review2
  };
  const archive2 = await runAgent(
    `你是归档 Agent。工作目录：${root}。1) 写 ${reviewPayload.file_path}：frontmatter（date/title/type/round/tags）+ 空行 + note_md；2) 写 ${reviewPayload.audit_path}（目录先 mkdir -p）；3) 运行 cd ${root} && python3 scripts/index.py，确认输出含 indexed 与 dashboard。数据：${JSON.stringify(reviewPayload, null, 2)}。返回 {ok, files, index_output}。`,
    { schema: archiveSchema, label: '归档' }
  );
  result.archive = archive2;
  if (!archive2 || !archive2.ok) {
    result.note = finalNote;
    result.review = review2;
    result.note_file = reviewPayload.file_path;
    result.note = '归档失败，返回原始内容由编排者手动落盘';
  }
}

if (mode === 'news') {
  phase('01 · 动态简报');
  log('动态记者正在整理本周值得知道的动态');
  const week = isoWeek(date);

  const newsPrompt = `你是《LLM Atlas》动态记者。今天 ${date}（本周：${week}）。

用 web_search / web_fetch 找出本周（最近 7 天）大模型行业最值得知道的 3–5 件事，面向正在系统打基础的工程师。

要求：
1. 与课程分开：这是资讯，不是课程正文；每件事写「发生了什么（2–3 句）→ 为什么值得知道（1 句）→ 与课程连接（1 句，说明它印证/挑战了正在学的哪个概念）→ 来源 URL」。
2. 与学习者相关优先：他正在按「推理优先、昇腾 + vLLM 主栈」的路径学习，推理框架、昇腾、vLLM、训练效率的动态优先。
3. 每条必须有可点击的来源 URL，不编造。
4. brief_md 结构：
# 动态简报 · ${week}（MM-DD ~ MM-DD）
> 本周值得知道的 N 件事，与课程分开，供碎片时间阅读。
## 1. 标题
- 发生了什么：……
- 为什么值得知道：……
- 与课程连接：……
- 来源：URL
（3–5 条）
5. 正文不超过 900 字。tags 2–4 个。只输出 JSON。`;

  const news = await runAgent(newsPrompt, { schema: newsSchema, label: '动态记者' });
  if (!news) throw new Error('动态记者未能完成简报');
  result.title = news.title;

  phase('02 · 归档');
  const newsPayload = {
    kind: 'news',
    file_path: `${root}/动态/${week}.md`,
    frontmatter: { date, title: news.title, type: 'news', week, tags: news.tags },
    note_md: news.brief_md
  };
  const archive3 = await runAgent(
    `你是归档 Agent。工作目录：${root}。1) 写 ${newsPayload.file_path}：frontmatter（date/title/type/week/tags）+ 空行 + note_md；2) 运行 cd ${root} && python3 scripts/index.py，确认输出含 indexed 与 dashboard。数据：${JSON.stringify(newsPayload, null, 2)}。返回 {ok, files, index_output}。`,
    { schema: archiveSchema, label: '归档' }
  );
  result.archive = archive3;
  if (!archive3 || !archive3.ok) {
    result.note = news;
    result.note_file = newsPayload.file_path;
    result.note = '归档失败，返回原始内容由编排者手动落盘';
  }
}

phase('完成');
log(`模式 ${mode} 完成`);
return result;
