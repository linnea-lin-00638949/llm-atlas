// LLM Atlas · 每日多 Agent 学习工作流
// 用法：由 DSH workflow 工具执行，args 传入 { date, domains, coverage, recent }
// 角色：策划 1 → 领域研究员 4（并行）→ 审校 1 → 主编 1

const A = args || {};
const date = A.date || new Date().toISOString().slice(0, 10);
const domains = A.domains || [];
const coverage = A.coverage || {};
const recent = A.recent || [];

const researchSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'brief_md', 'key_takeaways', 'terms', 'sources', 'open_questions', 'confidence', 'coverage_tags'],
  properties: {
    title: { type: 'string' },
    brief_md: { type: 'string' },
    key_takeaways: { type: 'array', items: { type: 'string' } },
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
        required: ['id', 'title', 'url', 'kind'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          url: { type: 'string' },
          published: { type: 'string' },
          kind: { type: 'string' }
        }
      }
    },
    open_questions: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    coverage_tags: { type: 'array', items: { type: 'string' } }
  }
};

const planSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['theme', 'thesis', 'domains'],
  properties: {
    theme: { type: 'string' },
    thesis: { type: 'string' },
    domains: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'module_id', 'topic', 'why_today', 'questions'],
        properties: {
          id: { type: 'string' },
          module_id: { type: 'string' },
          topic: { type: 'string' },
          why_today: { type: 'string' },
          questions: { type: 'array', items: { type: 'string' } }
        }
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
        required: ['domain_id', 'severity', 'claim', 'problem', 'fix'],
        properties: {
          domain_id: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          claim: { type: 'string' },
          problem: { type: 'string' },
          fix: { type: 'string' }
        }
      }
    }
  }
};

const digestSchema = {
  type: 'object',
  additionalProperties: false,
  required: ['title', 'digest_md', 'main_thread', 'tags', 'tomorrow_questions', 'knowledge_updates'],
  properties: {
    title: { type: 'string' },
    digest_md: { type: 'string' },
    main_thread: { type: 'string' },
    tags: { type: 'array', items: { type: 'string' } },
    tomorrow_questions: { type: 'array', items: { type: 'string' } },
    knowledge_updates: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['domain_id', 'module_id', 'concepts', 'next_focus'],
        properties: {
          domain_id: { type: 'string' },
          module_id: { type: 'string' },
          concepts: { type: 'array', items: { type: 'string' } },
          next_focus: { type: 'string' }
        }
      }
    }
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

function curriculumView(d) {
  return {
    id: d.id,
    name: d.name,
    mission: d.mission,
    watch: d.watch,
    curriculum: d.curriculum.map(m => ({ id: m.id, name: m.name, level: m.level, concepts: m.concepts }))
  };
}

function fallbackPlan() {
  return {
    theme: '推理成本的全栈拆解',
    thesis: '模型能力提升正在被推理成本重新定价；架构、引擎、硬件与训练数据的选择，最终都汇合到每个 token 的成本与延迟上。',
    domains: domains.map(d => {
      const learned = (coverage[d.id] && coverage[d.id].learned_modules) || [];
      const mod = d.curriculum.find(m => !learned.includes(m.id)) || d.curriculum[0];
      return {
        id: d.id,
        module_id: mod.id,
        topic: mod.name + '：今天的现实约束',
        why_today: '从成本与效率视角重新理解该模块。',
        questions: ['它解决什么问题？', '关键机制是什么？', '代价和边界在哪里？']
      };
    })
  };
}

phase('01 · 策划');
log('总策划正在选择今日跨领域主线与各领域切入点');

const planPrompt = `你是「LLM Atlas」学习系统的总策划。今天日期：${date}。

你的职责不是讲知识，而是为四个领域研究员安排今天唯一的一条学习主线，让四份简报能拼成一张完整的图，而不是四份互不相干的新闻。

学习者画像：${A.audience || '具备 Python 与深度学习基础、希望系统理解大模型全栈的工程师'}。

可用领域与课程模块：
${JSON.stringify(domains.map(curriculumView), null, 2)}

历史覆盖（可能为空）：
${JSON.stringify(coverage, null, 2)}

最近日报主题：
${JSON.stringify(recent, null, 2)}

要求：
1. 选一个能贯穿四个领域的今日主题（theme），例如「推理成本」「长上下文」「稀疏化」「Agent 的延迟预算」。它必须同时能落到模型、框架、硬件、训练四个视角。
2. 用一句话给出 thesis：今天要验证或推翻的核心判断。它应该有张力，不是正确的废话。
3. 为每个领域选择 1 个最贴题的课程模块（module_id 必须来自上面列表），并给出今天的具体切入点（topic），不要泛泛而谈。
4. 每个领域给 3 个递进问题：是什么 → 为什么这样设计 → 代价与边界。
5. 如果历史覆盖显示某模块已学，优先换未学模块；若必须复盘，在 why_today 中明确写「复盘 + 新证据」。
6. 四个领域必须真正互补：模型讲能力与架构，框架讲系统实现，硬件讲物理约束，训练讲能力从何而来。
7. 只输出 JSON。`;

const plan = (await runAgent(planPrompt, { schema: planSchema, label: '总策划' })) || fallbackPlan();
const planMap = {};
(plan.domains || []).forEach(p => { planMap[p.id] = p; });
log(`今日主线：${plan.theme}｜${plan.thesis}`);

phase('02 · 领域研究');
log('四位领域研究员并行检索、阅读与写作');

const researchPrompt = d => {
  const p = planMap[d.id] || fallbackPlan().domains.find(x => x.id === d.id);
  const mod = d.curriculum.find(m => m.id === p.module_id) || d.curriculum[0];
  return `你是「${d.name}」领域研究员，服务于一个跨领域的大模型学习系统。今天日期：${date}。

今日任务：${p.topic}
所属课程模块：${mod.name}（module_id: ${mod.id}）
模块核心概念：${mod.concepts.join(' / ')}
今日主线：${plan.theme}
主线判断：${plan.thesis}
为什么今天看这个：${p.why_today}
你需要回答的三个递进问题：
1. ${p.questions[0]}
2. ${p.questions[1]}
3. ${p.questions[2]}

工作方法（必须遵守）：
- 先 web_search 再 web_fetch。至少进行 6 次检索或抓取，中英文来源都要覆盖。
- 时间窗：优先最近 14 天的信息；背景知识可以更早，但必须标注时间，不要把旧闻写成新闻。
- 来源优先级：论文 / 官方文档 / release notes / 技术报告 > 高质量工程博客 > 新闻。至少 4 个来源，其中至少 2 个一手来源。
- 关键数字、版本号、日期、论文名后用 [S1] 形式标注，sources 里给出对应 url。每条 source 的 id 要与正文一致。
- 严格区分「已确认」「有争议」「我的推断」。推断必须显式写成“推断：……”。
- 禁止编造 URL、版本号、论文标题或 benchmark 数字。无法验证的内容删除或标注不确定。
- 不要写成新闻摘要。必须讲清机制：为什么快 / 为什么省 / 为什么难 / 代价是什么。
- 主动与另外三个领域连接：至少用一段说明你的发现如何影响模型、框架、硬件、训练中的另一个领域。

brief_md 使用以下结构（Markdown，不要加一级标题）：
## 今日问题
## 关键进展
## 原理拆解
## 工程取舍
## 与全栈的连接
## 还没想清楚

写作要求：中文，术语保留英文；1200–2000 字；信息密度高，删掉所有空话；每个小节都要有具体事实或机制，而不是“非常重要”“值得关注”。

只输出 JSON。`;
};

const briefResults = await parallel(
  domains.map(d => () =>
    runAgent(researchPrompt(d), {
      schema: researchSchema,
      label: `${d.emoji || ''}${d.name}研究员`
    })
  )
);

const briefs = {};
domains.forEach((d, i) => {
  const r = briefResults[i];
  if (r) briefs[d.id] = r;
  else briefs[d.id] = {
    title: `${d.name}：今日简报缺失`,
    brief_md: `## 今日问题\n\n研究员未能完成本次检索。\n\n## 关键进展\n\n无可靠信息。\n\n## 还没想清楚\n\n需要下次重试。`,
    key_takeaways: [],
    terms: [],
    sources: [],
    open_questions: [],
    confidence: 'low',
    coverage_tags: []
  };
});

const missing = domains.filter(d => !briefResults[domains.indexOf(d)]);
if (missing.length) log(`注意：${missing.map(d => d.name).join('、')} 简报缺失，已降级处理`);

phase('03 · 审校');
log('审校 Agent 正在核对事实、来源与跨领域一致性');

const reviewPrompt = `你是独立的审校 Agent。今天日期：${date}。下面是四位研究员为同一期学习日报提交的简报，以及今日计划。

今日计划：
${JSON.stringify(plan, null, 2)}

四份简报：
${JSON.stringify(domains.map(d => ({ domain_id: d.id, domain_name: d.name, brief: briefs[d.id] })), null, 2)}

逐项检查：
1. 事实：数字、版本号、日期、论文/产品名是否与来源一致；来源是否真的支持该结论；是否存在把厂商宣称当成独立事实。
2. 推理：是否把相关性当因果，是否跳过关键机制直接下结论。
3. 时效：是否把旧闻当新闻，是否遗漏明显更近的进展。
4. 跨领域：四份简报之间是否矛盾、重复，或漏掉了本该建立的连接。
5. 学习价值：是否只是罗列，没有机制解释和代价分析。

必要时用 web_fetch 核对最关键的一手来源。最多列 10 条问题，按严重程度排序。high = 事实错误或来源不支持，必须改；medium = 表述过强或缺少关键限定；low = 可改进。
如果整体可靠，verdict 为 pass；否则为 revise。只输出 JSON。`;

let review = await runAgent(reviewPrompt, { schema: reviewSchema, label: '审校' });
if (!review) {
  review = { verdict: 'pass', summary: '审校未返回结果，按未审校处理。', issues: [] };
}
log(`审校结论：${review.verdict}｜${(review.issues || []).length} 条问题`);

phase('04 · 汇总');
log('主编正在把四份简报重写成一份有主线的日报');

const editorPrompt = `你是《LLM Atlas 大模型学习日报》的主编。今天日期：${date}。

你的读者是一位有工程背景、想系统理解大模型全栈的学习者。他要的不是新闻列表，而是一条能记住、能迁移的因果链。

今日计划：
${JSON.stringify(plan, null, 2)}

四份领域简报：
${JSON.stringify(domains.map(d => ({ domain_id: d.id, domain_name: d.name, brief: briefs[d.id] })), null, 2)}

审校意见：
${JSON.stringify(review, null, 2)}

写作要求：
1. 先修复 high / medium 问题；无法修复的降级表述或删除。不要保留审校明确指出为错的 claim。
2. 这不是四篇简报的拼接。开头要用一条主线把四个领域串起来：同一个约束（成本、带宽、延迟、数据）如何在不同层表现。
3. 保留 [S1] 这类引用标记；文末按领域列出完整来源（标题 + URL + 日期）。
4. 中文写作，术语保留英文原名。删除“赋能”“范式革命”“值得关注”等空话。
5. 每个领域都要有“机制”和“代价”，不能只有结论。
6. 全文 2500–4000 字。

digest_md 必须严格使用以下结构：

# ${date} · 大模型学习日报

> 今日主线：一句话。

## 一分钟速览
- 4 条，每条不超过 40 字，必须包含具体对象或数字。

## 01 模型演进｜（小标题）
### 为什么今天看这个
### 关键进展
### 机制与取舍
### 一句话记住

## 02 推理框架｜（小标题）
（同样四小节）

## 03 硬件｜（小标题）
（同样四小节）

## 04 训练｜（小标题）
（同样四小节）

## 跨领域连接
- 3 条，每条写清 A → B 的因果，不要只说“密切相关”。

## 术语卡
| 术语 | 一句话解释 | 出现场景 |
| --- | --- | --- |
| ... | ... | ... |

## 自测三问
### 1. 问题
<details><summary>查看答案</summary>答案</details>
（共 3 题，覆盖至少 3 个领域）

## 明日待解
- 3 条，必须是可检索、可验证的问题。

## 来源
### 模型演进
- [S1] 标题 — URL（日期）
（四个领域都要列）

metadata 要求：
- main_thread：不超过 60 字的今日主线。
- tags：5–8 个可复用的知识标签。
- tomorrow_questions：3 条。
- knowledge_updates：每个领域一条，写清今天学了哪个 module、掌握了哪些概念、下一步看什么。

只输出 JSON。`;

const digest = await runAgent(editorPrompt, { schema: digestSchema, label: '主编' });
if (!digest) throw new Error('主编未能生成日报');

phase('05 · 归档');
log('归档 Agent 正在落盘并重建看板');

const root = A.root || '/Users/lyf02488571/Desktop/Learning';
const payload = { plan, briefs, review, digest };
const archivePrompt = `你是归档 Agent。工作目录（绝对路径）：${root}。日期：${date}。

用 write 工具写以下文件（目标目录不存在时先用 bash mkdir -p 创建；所有内容原样落盘，不要改写、不要加解释）：

1. 每个领域一个文件 ${root}/runs/${date}/<domain_id>.md：
   内容 = "# " + brief.title + "\\n\\n" + brief.brief_md

2. ${root}/runs/${date}/review.md，格式：
   # 审校记录 · ${date}
   **结论**：<verdict>
   **摘要**：<summary>
   ## 问题清单
   （每个 issue 一段：### <SEVERITY 大写> · <domain_id>，然后 - **claim**：… / - **问题**：… / - **建议**：…）
   没有问题时写「本期未发现需要修改的问题」。

3. ${root}/runs/${date}/plan.json = plan（JSON 原样）。

4. ${root}/daily/${date}.md：先写 frontmatter，再空一行，写 digest_md。frontmatter 用 JSON 值语法：
   ---
   date: "${date}"
   title: <digest.title>
   main_thread: <digest.main_thread>
   tags: <digest.tags 的 JSON 数组>
   domains: <briefs 所有键名的 JSON 数组>
   coverage: <由 digest.knowledge_updates 生成 {domain_id: [module_id]} 的 JSON 对象>
   generated: <当前北京时间 "YYYY-MM-DD HH:MM">
   ---

5. 用 bash 执行：cd ${root} && python3 scripts/index.py，确认输出里出现 indexed 与 dashboard 两行。失败就把错误信息带回。

完成后返回 JSON {ok: true, files: [写入的文件路径], index_output: "..."}。若某步失败，返回 {ok: false, error: "哪一步、什么错误"}。

以下是数据（不要修改内容）：
${JSON.stringify(payload, null, 2)}`;

const archive = await runAgent(
  archivePrompt,
  {
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['ok', 'files', 'index_output'],
      properties: {
        ok: { type: 'boolean' },
        files: { type: 'array', items: { type: 'string' } },
        index_output: { type: 'string' }
      }
    },
    label: '归档'
  }
);

phase('06 · 完成');
const out = {
  date,
  theme: plan.theme,
  title: digest.title,
  main_thread: digest.main_thread,
  review_verdict: review.verdict,
  review_issue_count: (review.issues || []).length,
  archive
};
if (!archive || !archive.ok) {
  out.briefs_compact = {};
  domains.forEach(d => {
    out.briefs_compact[d.id] = { title: briefs[d.id].title, brief_md: briefs[d.id].brief_md };
  });
  out.digest = digest;
  out.plan = plan;
  out.review = review;
  out.note = '归档失败，返回原始内容由编排者手动落盘';
}
log('完成');
return out;
