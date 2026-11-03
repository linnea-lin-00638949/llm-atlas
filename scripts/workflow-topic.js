// LLM Atlas · 专题工作流（独立于课程路径的知识专题）
// args: { date, config_path, root, topic, parts }
//   topic: { id, folder, name }
//   parts: [{ id, title, domain, focus[], questions[] }]
// 角色：策划 → 3 个研究员（并行）→ 助教审校 → 主编（并行修订）→ 归档

const A = args || {};
const date = A.date || new Date().toISOString().slice(0, 10);
const configPath = A.config_path || '/Users/lyf02488571/Desktop/Learning/config/agents.json';
const root = A.root || '/Users/lyf02488571/Desktop/Learning';
const topic = A.topic || { id: 'topic', folder: '专题', name: '专题' };
const parts = A.parts || [];

const planSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['moc_md', 'parts', 'reading_note'],
  properties: {
    moc_md: { type: 'string' },
    reading_note: { type: 'string' },
    parts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'focus', 'must_define', 'connect'],
        properties: {
          id: { type: 'string' },
          focus: { type: 'array', items: { type: 'string' } },
          must_define: { type: 'array', items: { type: 'string' } },
          connect: { type: 'string' }
        }
      }
    }
  }
};

const lessonBodySchema = {
  type: 'object',
  additionalProperties: false,
  required: ['note_md', 'terms', 'sources'],
  properties: {
    note_md: { type: 'string' },
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
        required: ['part_id', 'severity', 'claim', 'problem', 'fix'],
        properties: {
          part_id: { type: 'string' },
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
  required: ['title', 'note_md', 'tags'],
  properties: {
    title: { type: 'string' },
    note_md: { type: 'string' },
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

phase('01 · 策划');
log('专题策划正在定稿三个视角的教学提纲');

const plannerPrompt = `你是「LLM Atlas」的专题策划。今天 ${date}。

用 read 工具读取 ${configPath}：audience 是学习者画像，principles 是教学原则（地基优先、每个概念从零解释、不编造、存疑要标注）。

专题：${topic.name}（folder：${topic.folder}）
三个视角的草稿提纲：
${JSON.stringify(parts, null, 2)}

任务：
1. 为每个 part 定稿 focus（3–5 个必须讲透的点，保持草稿方向但可修正）；
2. 给每个 part 列 must_define（该篇第一次出现就必须解释的术语，至少 6 个）；
3. 给每个 part 写 connect（一段话：这篇与其他两篇、与课程已学内容的关系）；
4. 写 moc_md：专题总览笔记正文（Markdown，不超 500 字）：
   # ${topic.name}（专题）
   > 一句话说明这个专题解决什么问题
   ## 为什么学
   ## 三篇导读（每篇 1–2 句）
   ## 建议阅读顺序
   ## 与课程的关系（引用已学课程概念，如「第 1 课的前向传播」）
5. 写 reading_note：给读者的学习建议（1–2 句）。

读者基础弱：SSM/状态空间这类概念必须从零讲，与 Transformer 对比着讲。只输出 JSON。`;

const plan = await runAgent(plannerPrompt, { schema: planSchema, label: '专题策划' });
if (!plan) throw new Error('专题策划失败');
const partPlan = {};
(plan.parts || []).forEach(p => { partPlan[p.id] = p; });

phase('02 · 研究');
log('三个研究员并行调研与写作');

const researchPrompt = (part) => {
  const p = partPlan[part.id] || {};
  return `你是「${topic.name}」专题的研究员，负责「${part.title}」这一篇。今天 ${date}。

用 read 工具读取 ${configPath} 了解学习者画像（基础弱、用昇腾 + vLLM）与教学原则。

这篇必须讲透：
${(p.focus || part.focus || []).map((f, i) => `${i + 1}. ${f}`).join('\n')}

这些术语第一次出现必须解释（用白话 + 直觉类比，再给严格定义）：
${(p.must_define || []).join('、')}

与其他两篇及课程的关系：${p.connect || ''}

研究要求：
1. 用 web_search / web_fetch 查证事实与最新支持情况（论文、官方文档、GitHub issue/release 为主，厂商宣传材料要交叉验证）。重点线索：
   - GDN = Gated Delta Networks（ICLR 2025，Mamba-2 + delta rule，arXiv:2412.06464），以论文为准；
   - KDA = Kimi Delta Attention（Moonshot Kimi Linear 的硬件感知 delta attention），以官方技术博客/论文为准，查不到就明确写「未能核实」；
   - vLLM 对 Mamba/Jamba 等混合模型的支持现状、所用 kernel（mamba_ssm / causal-conv1d）、V1 支持历史、投机解码与缓存行为；
   - vllm-ascend 对 Mamba/GDN 的支持与昇腾上的性能特点（如 GitHub 上的 vllm-ascend 相关仓库与 issue）。
2. 每个关键结论后给来源 URL；sources 至少 4 条，其中至少 2 条一手来源。
3. 面向基础弱的读者：所有概念从零讲；与 Transformer 对比（读者熟悉 attention 的直觉）；带具体数字的例子；常见误区至少 2 个。
4. note_md 结构（不要一级标题）：
## 为什么需要它
## 核心概念
### <概念名>
（定义 → 直觉 → 机制）
## 一步一步理解
## 常见误区
## 一句话总结
## 术语卡
| 术语 | 解释 |
| --- | --- |
## 还没想清楚
## 来源
5. 中文，1800–3000 字；「存疑」与「已确认」分开；不编造版本号与数字。
6. terms 数组与术语卡一致。

只输出 JSON。`;
};

const briefResults = await parallel(
  parts.map(part => () =>
    runAgent(researchPrompt(part), { schema: lessonBodySchema, label: `研究员·${part.title}` })
  )
);
const briefs = {};
parts.forEach((part, i) => {
  briefs[part.id] = briefResults[i] || {
    note_md: `## 为什么需要它\n\n本篇未能完成调研，需要重试。`,
    terms: [],
    sources: []
  };
});

phase('03 · 审校');
log('助教正在核对三篇的事实与初学者友好度');

const auditPrompt = `你是助教。审校「${topic.name}」专题的三篇笔记。逐篇检查：概念与数学是否正确；术语是否都解释了；有没有把厂商宣称/推测写成事实；来源是否支持结论；三篇之间是否矛盾或重复；对基础弱的读者是否可读。最多 10 条问题，每条标 part_id 与 severity（high=事实错误必须改；medium=难懂或缺限定；low=可改进）。只输出 JSON。

三篇笔记：
${JSON.stringify(parts.map(part => ({ part_id: part.id, title: part.title, body: briefs[part.id] })), null, 2)}`;

let review = await runAgent(auditPrompt, { schema: reviewSchema, label: '助教' });
if (!review) review = { verdict: 'pass', summary: '审校未返回结果。', issues: [] };
log(`审校结论：${review.verdict}｜${(review.issues || []).length} 条问题`);

phase('04 · 成稿');
log('主编正在按审校意见修订三篇笔记');

const editorPrompt = (part) => {
  const issues = (review.issues || []).filter(i => i.part_id === part.id);
  return `你是《LLM Atlas》主编。修订「${topic.name}」专题的「${part.title}」一篇。

先修复审校指出的 high / medium 问题，无法修复的降级表述或删除；不要把「存疑」写成事实。

审校意见（只针对本篇）：
${JSON.stringify(issues, null, 2)}

原始笔记：
${JSON.stringify(briefs[part.id], null, 2)}

note_md 结构（一级标题用 title）：

# ${part.title}

> 读完这篇，你能：…（2–3 条）

## 为什么需要它
## 核心概念
## 一步一步理解
## 常见误区
## 一句话总结
## 自测题
### 1. 题目
<details><summary>查看答案</summary>答案</details>
（3–4 题，答案详细）
## 术语卡
| 术语 | 解释 |
| --- | --- |
## 还没想清楚
## 来源

中文，1800–3000 字；面向基础弱的读者；tags 4–6 个。只输出 JSON。`;
};

const finalResults = await parallel(
  parts.map(part => () =>
    runAgent(editorPrompt(part), { schema: noteSchema, label: `主编·${part.title}` })
  )
);
const finals = {};
parts.forEach((part, i) => {
  finals[part.id] = finalResults[i];
});
if (parts.some(part => !finals[part.id])) throw new Error('主编未能完成修订');

phase('05 · 归档');
log('归档 Agent 正在落盘并重建索引');

const archivePayload = {
  kind: 'topic',
  topic_id: topic.id,
  folder: topic.folder,
  name: topic.name,
  date,
  moc_path: `${root}/专题/${topic.folder}/${topic.folder}.md`,
  moc_md: plan.moc_md,
  reading_note: plan.reading_note,
  parts: parts.map((part, i) => ({
    id: part.id,
    title: part.title,
    order: i + 1,
    file_path: `${root}/专题/${topic.folder}/${part.id}.md`,
    note_md: finals[part.id].note_md,
    tags: finals[part.id].tags
  })),
  audit_path: `${root}/runs/专题-${topic.id}-${date}/review.md`,
  plan_path: `${root}/runs/专题-${topic.id}-${date}/plan.json`,
  review
};

const mocLinks = parts
  .map(part => `[[专题/${topic.folder}/${part.id}|${part.title}]]`)
  .join('\n');

const archivePrompt = `你是归档 Agent。工作目录：${root}。日期：${date}。

任务（内容原样落盘，不要改写）：
1. mkdir -p ${root}/专题/${topic.folder}；
2. 写专题总览 ${archivePayload.moc_path}，frontmatter：
   date / title（「${topic.name}（专题）」）/ type: "topic-index" / topic_id / tags
   然后空一行写 moc_md，再写「## 三篇笔记」列出三个 part 的 wikilink：
   ${mocLinks}
3. 每个 part 写 ${root}/专题/${topic.folder}/<id>.md，frontmatter：
   date / title / type: "topic" / topic（专题名）/ topic_id / part（part.id）/ part_title / order / tags
   然后空一行写 note_md。
4. 写 ${archivePayload.audit_path}（mkdir -p 目录）与 ${archivePayload.plan_path}。
4. 绝对不要移动、重命名或删除 ${root}/课程、${root}/runs、${root}/专题 下的任何已有文件；本任务只允许新建/覆盖上面指定的路径。发现疑似临时文件也不要动，报告给编排者。

5. bash 执行 cd ${root} && python3 scripts/index.py，确认输出含 indexed 与 dashboard。

数据：
${JSON.stringify(archivePayload, null, 2)}

返回 {ok, files, index_output}。`;

const archive = await runAgent(archivePrompt, { schema: archiveSchema, label: '归档' });

const out = {
  date,
  topic_id: topic.id,
  topic: topic.name,
  review_verdict: review.verdict,
  review_issue_count: (review.issues || []).length,
  archive
};
if (!archive || !archive.ok) {
  out.payload = archivePayload;
  out.note = '归档失败，返回原始内容由编排者手动落盘';
}
phase('完成');
return out;
