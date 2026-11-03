// LLM Atlas v2 · 批量课程生成工作流
// 每门课独立流水线：讲师 → 助教 → 主编 → 归档（pipeline 无屏障并行）
// 批次结束后，按轮次串讲复习（读已落盘的课程文件）
// args: { date, root, audience, modules: [...], rounds: [...] }

const A = args || {};
const date = A.date;
const root = A.root || '/Users/lyf02488571/Desktop/Learning';
const audience = A.audience || '刚入门、基础较弱的工程师';
const modules = A.modules || [];
const rounds = A.rounds || [];

function lessonTitle(m) {
  let t = `第 ${m.lesson_no} 课 · ${m.name}`;
  if (m.days > 1) t += m.part === 1 ? '（上）' : '（下）';
  return t;
}

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

const issueItem = {
  type: 'object',
  additionalProperties: false,
  required: ['severity', 'claim', 'problem', 'fix'],
  properties: {
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
    claim: { type: 'string' },
    problem: { type: 'string' },
    fix: { type: 'string' }
  }
};

const reviewObj = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'summary', 'issues'],
  properties: {
    verdict: { type: 'string', enum: ['pass', 'revise'] },
    summary: { type: 'string' },
    issues: { type: 'array', items: issueItem }
  }
};

const auditOutSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['lesson_md', 'terms', 'sources', 'review'],
  properties: {
    lesson_md: { type: 'string' },
    terms: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['term', 'explanation'], properties: { term: { type: 'string' }, explanation: { type: 'string' } } } },
    sources: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['title', 'url', 'kind'], properties: { title: { type: 'string' }, url: { type: 'string' }, kind: { type: 'string' } } } },
    review: reviewObj
  }
};

const noteSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'note_md', 'goals', 'tags', 'review'],
  properties: {
    title: { type: 'string' },
    note_md: { type: 'string' },
    goals: { type: 'array', items: { type: 'string' } },
    tags: { type: 'array', items: { type: 'string' } },
    review: reviewObj
  }
};

const archiveSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['ok', 'file', 'title'],
  properties: {
    ok: { type: 'boolean' },
    file: { type: 'string' },
    title: { type: 'string' },
    issue_count: { type: 'integer' }
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

phase('备课');
log(`批量备课：${modules.length} 门课并行流水线（讲师 → 助教 → 主编 → 归档）`);

async function teach(prev, m) {
  const title = lessonTitle(m);
  const prompt = `你是「${m.folder}」课程的讲师。今天 ${date}，${title}。

学习者画像：${audience}。目标是把地基打牢，不追新闻。

模块信息：
${JSON.stringify(m, null, 2)}

${m.part_note ? `本课分集说明（本集要讲的部分）：${m.part_note}\n` : ''}

备课要求：
1. 来源只用稳定材料：sources_hint 里的材料、官方文档、经典论文、公认教材。禁止把「最近两周的新闻」当正文。sources 每条给出 title / url / kind（官方文档/论文/教材/教程）。
2. 每个术语第一次出现必须解释；不用未定义的英文缩写；复杂概念先直觉类比、再严格定义。
3. 必须有一个逐步推导的 worked example，带具体数字，读者能跟着手算。
4. 「常见误区」至少 2 个，指出错误直觉并纠正。
5. 前置模块 ${JSON.stringify(m.prereqs || [])} 与你在同一批次生成、可能还没写完：用到它的概念时，先一句话回顾再使用，不要假设读者记得细节。
6. 能落到「昇腾 NPU 上跑 vLLM」的工程实践就落（不要硬凑）。
7. lesson_md 结构（Markdown，不要一级标题）：
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
8. 中文，1800–3000 字；删除空话；公式写清楚；不确定写「存疑」，不要编造。
9. terms 数组与术语卡表格一致。

只输出 JSON。`;
  return await runAgent(prompt, { schema: lessonBodySchema, label: `${title} · 讲师` });
}

async function audit(lesson, m) {
  if (!lesson) return null;
  const title = lessonTitle(m);
  const prompt = `你是助教。今天 ${date}，审校课程「${title}」（模块 ${m.name}，模块目标：${JSON.stringify(m.goals)}）。

这份讲义是给基础较弱的工程师看的。逐项检查：
1. 概念与数学是否正确（重点：公式、worked example 的每个数字要重算）。
2. 初学者友好度：有没有没解释就用的术语/缩写？例子能跟着手算吗？
3. 有没有把「存疑/推测」写成事实？
4. 是否偷偷假设了还没学的知识？
5. 教学结构是否完整：课前自测、核心概念、例子、误区、一句话总结、术语卡、下一步、来源？
6. 是否覆盖了模块 goals 里的要点？

讲义全文：
${JSON.stringify(lesson, null, 2)}

最多列 8 条问题，按严重程度排序（high = 概念错误必须改；medium = 难懂或缺关键限定；low = 可改进）。没有问题就 verdict: pass、issues 为空。
输出 = 讲义原样 + review 字段。只输出 JSON。`;
  return await runAgent(prompt, { schema: auditOutSchema, label: `${title} · 助教` });
}

async function edit(audited, m) {
  if (!audited) return null;
  const title = lessonTitle(m);
  const prompt = `你是《LLM Atlas》课程主编。今天 ${date}，${title}。

先修复审校指出的 high / medium 问题，无法修复的降级表述或删除；不要把「存疑」写成事实。

模块信息：${JSON.stringify(m, null, 2)}
讲义与审校意见：
${JSON.stringify(audited, null, 2)}

note_md 必须严格使用以下结构（一级标题用 title）：

# ${title}

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
（一句话说明下一课会讲什么${m.next_up ? '：' + m.next_up : ''}）

## 来源
（逐条列 title — URL）

写作要求：中文，术语保留英文；1800–3000 字；信息密度高；每个概念都有直觉、机制、例子三层。tags 给 4–6 个可复用标签。
输出字段：title / note_md / goals（3 条，对应开头的「学完今天，你能」）/ tags / review（审校意见原样带出）。
只输出 JSON。`;
  return await runAgent(prompt, { schema: noteSchema, label: `${title} · 主编` });
}

async function archive(note, m) {
  if (!note || !note.note_md) return null;
  const title = lessonTitle(m);
  const file = `${root}/课程/${m.folder}/${m.filename}.md`;
  const reviewFile = `${root}/runs/${date}/review-${m.filename}.md`;
  const prompt = `你是归档 Agent。工作目录：${root}。日期：${date}。

任务（内容原样落盘，不要改写）：
1. 用 bash mkdir -p 确保目录存在，然后用 write 工具写课程笔记 ${file}：
   先写 frontmatter（三个短横线起止，值用 JSON 语法）：
   date: "${date}"
   title: <title>
   domain: "${m.folder}"
   module_id: "${m.module_id}"
   module: "${m.name}"
   status: "已学"
   lesson_no: ${m.lesson_no}
   goals: <goals 的 JSON 数组>
   tags: <tags 的 JSON 数组>
   然后空一行，写 note_md。
2. 写审校记录 ${reviewFile}：含 verdict / summary / issues（每条：severity、claim、problem、fix）。
3. 不要运行 index.py。
4. 绝对不要移动、重命名或删除其他任何文件；只允许新建/覆盖上面指定的两个路径。

数据：
${JSON.stringify({ title, frontmatter: { date, title, domain: m.folder, module_id: m.module_id, module: m.name, status: '已学', lesson_no: m.lesson_no, goals: note.goals, tags: note.tags }, note_md: note.note_md, review: note.review }, null, 2)}

返回 {ok: true, file, title, issue_count: (note.review && note.review.issues ? note.review.issues.length : 0)}。失败返回 {ok: false, ...} 并说明。`;
  const r = await runAgent(prompt, { schema: archiveSchema, label: `${title} · 归档` });
  if (r) return r;
  return { ok: false, file, title, error: '归档未返回' };
}

const results = await pipeline(modules, teach, audit, edit, archive);

const items = [];
modules.forEach((m, i) => {
  const r = results[i];
  items.push({
    module_id: m.module_id,
    lesson_no: m.lesson_no,
    title: lessonTitle(m),
    file: r ? r.file : null,
    ok: !!(r && r.ok),
    issue_count: r && r.issue_count != null ? r.issue_count : null,
    verdict: r && r.verdict ? r.verdict : null
  });
});
const failed = items.filter(x => !x.ok);
log(`批次完成：${items.length - failed.length}/${items.length} 门课落盘${failed.length ? '，失败：' + failed.map(f => f.title).join('、') : ''}`);

const reviewResults = [];
if (rounds.length) {
  phase('串讲');
  log(`生成 ${rounds.length} 轮串讲复习`);
  await parallel(rounds.map(r => async () => {
    const title = `第 ${r.round} 轮串讲 · ${date}`;
    const prompt = `你是《LLM Atlas》串讲师。今天 ${date}，第 ${r.round} 轮串讲复习。

第一步：用 read 工具依次读完本轮四课：
${r.files.map((f, i) => `${i + 1}. ${f}`).join('\n')}
第二步：写串讲笔记 ${root}/课程/串讲/${date}-第${r.round}轮串讲.md，先写 frontmatter：
date: "${date}"
title: "${title}"
type: "review"
round: ${r.round}
tags: ["串讲", "复习"]
然后空一行写正文，结构：

# ${title}

> 本轮四课：${(r.titles || []).join('、')}

## 一张图串起来
（依赖关系/数据流/因果链把四课串起来，可用文字图）

## 模块速览表
| 课 | 一句话核心 | 关键公式/概念 | 与其他课的关系 |

## 综合自测
### 1. 题目
<details><summary>查看答案</summary>答案</details>
（6–8 题，跨课综合，答案详细）

## 易混淆对照表
| 容易混的概念 | 区别一句话 |

## 自查问题
（3 个问题，答不上来就回看对应课程）

## 下一轮预告

写作要求：中文，1500–2500 字；面向基础弱的读者；不要运行 index.py。
返回 {ok: true, file: "${root}/课程/串讲/${date}-第${r.round}轮串讲.md", title: "${title}"}。`;

    const res = await runAgent(prompt, {
      schema: { type: 'object', additionalProperties: false, required: ['ok', 'file', 'title'], properties: { ok: { type: 'boolean' }, file: { type: 'string' }, title: { type: 'string' } } },
      label: `第 ${r.round} 轮串讲`
    });
    reviewResults.push(res || { ok: false, round: r.round, title });
  }));
}

phase('完成');
log('批量工作流结束，等待编排者重建索引');
return { date, items, failed: failed.map(f => f.title), reviews: reviewResults };
