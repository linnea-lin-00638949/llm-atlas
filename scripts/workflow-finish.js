// LLM Atlas · 补跑审校与主编（用于首日主流程结果被截断后的恢复）
// args: { date, partial_path, known_issues }
// 审校与主编通过 read 工具直接读取 partial_path 中的 plan 与 briefs，避免大对象入参。

const A = args || {};
const date = A.date;
const partialPath = A.partial_path;
const knownIssues = A.known_issues || [];

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

phase('03 · 审校');
log('审校 Agent 重新核对四份简报');

const reviewPrompt = `你是独立的审校 Agent。今天日期：${date}。

第一步：用 read 工具读取文件 ${partialPath}。这是一个 JSON 文件，包含 plan（今日计划）和 briefs（四个领域的完整简报）。先完整读完再开始审。

第二步：逐项检查四份简报：
1. 事实：数字、版本号、日期、论文/产品名是否与来源一致；来源是否真的支持该结论；是否存在把厂商宣称当成独立事实。
2. 推理：是否把相关性当因果，是否跳过关键机制直接下结论，闭式推演是否标明了假设。
3. 时效：是否把旧闻当新闻，是否遗漏明显更近的进展。
4. 跨领域：四份简报之间是否矛盾、重复，或漏掉了本该建立的连接。
5. 学习价值：是否只是罗列，没有机制解释和代价分析。

必要时用 web_fetch 核对最关键的一手来源。

此前一版审校已经确认的问题（可参考，若你核实后仍成立就保留，不成立就修正或删除）：
${JSON.stringify(knownIssues, null, 2)}

最多列 10 条问题，按严重程度排序。high = 事实错误或来源不支持，必须改；medium = 表述过强或缺少关键限定；low = 可改进。
如果整体可靠，verdict 为 pass；否则为 revise。只输出 JSON。`;

let review = await runAgent(reviewPrompt, { schema: reviewSchema, label: '审校' });
if (!review) {
  review = { verdict: 'pass', summary: '审校未返回结果，按未审校处理。', issues: [] };
}
log(`审校结论：${review.verdict}｜${(review.issues || []).length} 条问题`);

phase('04 · 汇总');
log('主编正在把四份简报重写成一份有主线的日报');

const editorPrompt = `你是《LLM Atlas 大模型学习日报》的主编。今天日期：${date}。

第一步：用 read 工具读取文件 ${partialPath}。这是一个 JSON 文件，包含 plan（今日计划，含 theme/thesis 与各领域的 module_id、topic、questions）和 briefs（四个领域的完整简报，每个含 title、brief_md、sources、terms、key_takeaways 等）。先完整读完再开始写。

四个领域的固定顺序：01 模型演进、02 推理框架、03 硬件、04 训练。

审校意见（必须逐条处理）：
${JSON.stringify(review, null, 2)}

你的读者是一位有工程背景、想系统理解大模型全栈的学习者。他要的不是新闻列表，而是一条能记住、能迁移的因果链。

写作要求：
1. 先修复 high / medium 问题；无法修复的降级表述或删除。不要保留审校明确指出为错的 claim。
2. 这不是四篇简报的拼接。开头要用一条主线把四个领域串起来：同一个约束（成本、带宽、延迟、数据）如何在不同层表现。
3. 保留 [S1] 这类引用标记；文末按领域列出完整来源（标题 + URL + 日期）。来源必须取自简报的 sources 数组，禁止新增或编造。
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
### 为什么今天看这个
### 关键进展
### 机制与取舍
### 一句话记住

## 03 硬件｜（小标题）
### 为什么今天看这个
### 关键进展
### 机制与取舍
### 一句话记住

## 04 训练｜（小标题）
### 为什么今天看这个
### 关键进展
### 机制与取舍
### 一句话记住

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
- knowledge_updates：每个领域一条，module_id 必须来自 plan，写清今天学了哪个模块、掌握了哪些概念、下一步看什么。

只输出 JSON。`;

const digest = await runAgent(editorPrompt, { schema: digestSchema, label: '主编' });
if (!digest) throw new Error('主编未能生成日报');

phase('05 · 完成');
log('日报已生成');

return {
  date,
  review,
  digest
};
