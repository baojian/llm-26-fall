/* Build the reading page from Markdown and render editable Mermaid sources to SVG. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(root, 'slides/package.json'));
const MarkdownIt = require('markdown-it');
const { chromium } = require('playwright');
const slug = 'lecture-01-pre-tokenization';
const diagramRoot = path.join(root, 'docs/diagrams/pre-tokenization');
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const makeSlug = text => text.toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu, '').replaceAll(' ', '-');
const bilingual = (en, zh, tag = 'span') => `<${tag} data-en="${escape(en)}" data-zh="${escape(zh)}">${escape(en)}</${tag}>`;
const sources = {};
for (const language of ['en', 'zh']) {
  const suffix = language === 'zh' ? '.zh' : '';
  sources[language] = await readFile(path.join(root, `docs/${slug}${suffix}.md`), 'utf8');
}

export function renderArticle(source, language, canonicalHeadings) {
  const md = new MarkdownIt({ html: false, linkify: false, typographer: false });
  const tokens = md.parse(source, {});
  const headings = [];
  let index = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open') continue;
    const title = tokens[i + 1].content;
    const key = canonicalHeadings ? canonicalHeadings[index].key : makeSlug(title);
    const heading = { title, key, level: Number(tokens[i].tag.slice(1)) };
    if (canonicalHeadings) {
      assert.equal(heading.level, canonicalHeadings[index].level, `Heading level mismatch: ${title}`);
      assert.equal(title.match(/^\d+(?:\.\d+)*/)?.[0], canonicalHeadings[index].title.match(/^\d+(?:\.\d+)*/)?.[0], `Section mismatch: ${title}`);
    }
    headings.push(heading);
    tokens[i].attrSet('id', `${language}-${key}`);
    tokens[i].attrSet('data-section', key);
    index++;
  }
  if (canonicalHeadings) assert.equal(headings.length, canonicalHeadings.length, 'Translation is missing headings.');
  const inline = md.renderer.rules.link_open || ((items, i, options, env, self) => self.renderToken(items, i, options));
  md.renderer.rules.link_open = (items, i, options, env, self) => {
    const token = items[i];
    const href = token.attrGet('href');
    if (href?.startsWith('#')) token.attrSet('href', `#${language}-${href.slice(1)}`);
    return inline(items, i, options, env, self);
  };
  md.renderer.rules.table_open = () => '<div class="table-scroll" tabindex="0"><table>\n';
  md.renderer.rules.table_close = () => '</table></div>\n';
  const fence = md.renderer.rules.fence;
  md.renderer.rules.fence = (items, i, options, env, self) => `<div class="code-block"><button class="copy-code" type="button" data-en="Copy" data-zh="复制">Copy</button>${fence(items, i, options, env, self)}</div>`;
  // The page shell supplies the title and its accessible sidebar supplies the TOC.
  const filtered = [];
  let skipping = false;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type === 'heading_open' && tokens[i].tag === 'h1') { i += 2; continue; }
    if (tokens[i].type === 'heading_open' && tokens[i].tag === 'h2') {
      skipping = !/^\d+\./.test(tokens[i + 1].content);
    }
    if (!skipping) filtered.push(tokens[i]);
  }
  return { html: md.renderer.render(filtered, md.options, {}), headings };
}

const english = renderArticle(sources.en, 'en');
const chinese = renderArticle(sources.zh, 'zh', english.headings);
// Translations must keep executable code, sample data, and source references intact.
const codeBlocks = source => [...source.matchAll(/^```(python|bash|json)\n([\s\S]*?)^```/gm)].map(match => match[0]);
assert.deepEqual(codeBlocks(sources.zh), codeBlocks(sources.en), 'Translated code or sample data differs.');
const references = source => [...source.matchAll(/^\[([^\]]+)\]: (.+)$/gm)].map(match => [match[1], match[2]]);
assert.deepEqual(references(sources.zh), references(sources.en), 'Translation references differ.');

const diagrams = {};
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  await page.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body></body></html>');
  await page.addScriptTag({ path: path.join(root, 'slides/node_modules/mermaid/dist/mermaid.min.js') });
  await page.evaluate(() => mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    htmlLabels: false,
    theme: 'base',
    look: 'classic',
    deterministicIds: true,
    fontFamily: 'Arial, PingFang SC, Microsoft YaHei, sans-serif',
    themeVariables: {
      fontSize: '18px', primaryColor: '#e8f0fa', primaryTextColor: '#142e4b',
      primaryBorderColor: '#386998', lineColor: '#718497', clusterBkg: '#fbfcfe',
      clusterBorder: '#cdd8e3', edgeLabelBackground: '#ffffff',
    },
    flowchart: { look: 'classic', useMaxWidth: false, wrappingWidth: 280, curve: 'linear', nodeSpacing: 25, rankSpacing: 35, padding: 15 },
  }));
  for (const language of ['en', 'zh']) {
    for (const view of ['corpus', 'learning', 'encoding']) {
      const filename = `${view}.${language}`;
      const source = await readFile(path.join(diagramRoot, `${filename}.mmd`), 'utf8');
      const svg = await page.evaluate(async ({ id, source }) => (await mermaid.render(id, source)).svg, { id: `pipeline-${view}-${language}`, source });
      assert.ok(svg.includes('<svg') && !svg.includes('<foreignObject'), `Expected SVG text labels: ${filename}`);
      diagrams[filename] = svg;
      await writeFile(path.join(diagramRoot, `${filename}.svg`), `${svg}\n`);
    }
  }
} finally { await browser?.close(); }

const descriptions = {
  en: {
    corpus: 'A common corpus-preparation design. Extraction branches by source; filtering, deduplication, and exclusions can run in multiple passes. The output supplies a tokenizer-learning sample and the later LLM-training mixture.',
    learning: 'The learner sees a selected sample through the configured text pipeline. Vocabulary learning ends before the neural language model is trained. Hold-out evaluation can lead to revised sampling or segmentation choices.',
    encoding: 'Prepared training text and application input reuse the frozen tokenizer. Corpus-wide selection and deduplication do not run on every prompt. Chat formatting, special-token handling, and training-only augmentation follow the target model.',
  },
  zh: {
    corpus: '一种常见的语料准备设计。提取按来源分支，筛选、去重与排除可以多轮进行。整理后的语料分别供分词器抽样学习，以及后续 LLM 训练配比使用。',
    learning: '学习器通过配置好的文本处理流程读取选定样本。词表学习先于神经语言模型训练完成。留出集评估可能促使我们修改抽样或切分方案。',
    encoding: '准备好的训练文本与应用输入复用冻结的分词器。每条提示词不会重新执行全语料筛选与去重。对话格式、特殊 token 和训练专用增强须遵循目标模型。',
  },
};
const controls = `<div class="lang-toggle" role="group" aria-label="Language / 语言"><button type="button" data-lang="en" lang="en" aria-pressed="true">EN</button><button type="button" data-lang="zh" lang="zh-CN" aria-pressed="false">中文</button></div>`;
const panels = ['en', 'zh'].map(language => ['corpus', 'learning', 'encoding'].map(view => {
  const active = language === 'en' && view === 'corpus';
  const section = view === 'corpus' ? '3-a-practical-corpus-preparation-workflow' : view === 'learning' ? '4-the-preprocessing-inside-a-tokenizer' : '5-which-steps-repeat-during-encoding';
  return `<section class="pipeline-panel" id="panel-${view}-${language}" role="tabpanel" aria-labelledby="tab-${view}" data-view="${view}" data-language="${language}"${active ? '' : ' hidden'}>
<div class="diagram-scroll" tabindex="0" data-view="${view}" aria-label="${language === 'zh' ? '流程图，可横向滚动查看' : 'Pipeline diagram; scroll horizontally if needed'}">${diagrams[`${view}.${language}`]}</div>
<p class="diagram-caption">${descriptions[language][view]}</p>
<div class="diagram-actions"><a href="#${language}-${section}">${language === 'zh' ? '阅读步骤详解 ↓' : 'Read the detailed steps ↓'}</a><a href="diagrams/pre-tokenization/${view}.${language}.svg" download>${language === 'zh' ? '下载 SVG' : 'Download SVG'}</a><a href="diagrams/pre-tokenization/${view}.${language}.mmd">${language === 'zh' ? 'Mermaid 源码' : 'Mermaid source'}</a></div>
</section>`;
}).join('\n')).join('\n');
const nav = (article, language) => `<ol data-toc-language="${language}"${language === 'zh' ? ' hidden' : ''}>${article.headings.filter(h => h.level === 2 && /^\d+\./.test(h.title)).map(h => `<li><a href="#${language}-${h.key}" data-section="${h.key}">${escape(h.title)}</a></li>`).join('\n')}</ol>`;
const html = `<!doctype html>
<!-- Generated by scripts/build-notes.mjs. Edit the Markdown and Mermaid sources. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text preprocessing before tokenization · Lecture 01 · CS40008.01</title>
<meta name="description" content="Practical LLM text preprocessing: corpus preparation, tokenizer learning, and encoding, with Kimi K3, GLM-5.2, and DeepSeek-V4 evidence.">
<link rel="stylesheet" href="../assets/notes.css">
<link rel="stylesheet" href="../assets/course-language.css">
<script src="../assets/course-language.js" defer></script>
<script src="../assets/notes.js" defer></script>
</head>
<body>
<a href="#reading-content" class="skip-link">${bilingual('Skip to the notes', '跳转至讲义正文')}</a>
<header class="reading-header"><nav aria-label="Reading tools">
<a class="course-home" href="../index.html">CS40008.01 · ${bilingual('Course', '课程主页')}</a>
<a href="../slides/lecture-01/">${bilingual('Lecture 01 slides', '第 1 讲课件')}</a>
<button class="print-note" type="button">${bilingual('Print', '打印')}</button>
${controls}
</nav></header>
<div class="reading-layout">
<aside class="reading-sidebar"><details open><summary>${bilingual('On this page', '本页目录')}</summary>
<p class="sidebar-label">${bilingual('Lecture 01 · Reading', '第 1 讲 · 阅读材料')}</p>
<a class="pipeline-jump" href="#pipeline">${bilingual('Explore the pipeline', '查看处理流程')}</a>
${nav(english, 'en')}${nav(chinese, 'zh')}
<p class="source-links">${bilingual('Markdown source', 'Markdown 源文件')}<br><a href="${slug}.md" lang="en">English</a> · <a href="${slug}.zh.md" lang="zh-CN">中文</a></p>
</details></aside>
<main class="reading-main" id="reading-content">
<p class="reading-kicker">${bilingual('TEXT → TOKENS', '文本 → TOKEN')}</p>
<h1 class="reading-title">${bilingual('Text preprocessing before tokenization', '分词之前的文本预处理')}</h1>
<p class="reading-intro">${bilingual('From raw documents to a reusable tokenizer: practical steps, model evidence, and the boundary between training and inference.', '从原始文档到可复用的分词器：实际处理步骤、模型公开证据，以及训练与推理的边界。')}</p>
<section class="pipeline" id="pipeline" aria-labelledby="pipeline-title">
<div class="pipeline-heading"><h2 id="pipeline-title">${bilingual('Three stages, different responsibilities', '三个阶段，各有职责')}</h2><a class="diagram-credit" href="https://mermaid.js.org/">Mermaid · SVG</a></div>
<div class="pipeline-tabs" role="tablist" aria-label="Pipeline stage / 流程阶段">
${[['corpus', '1 · Prepare the corpus', '1 · 准备语料'], ['learning', '2 · Learn the tokenizer', '2 · 学习分词器'], ['encoding', '3 · Encode text', '3 · 编码文本']].map(([view, en, zh], i) => `<button type="button" role="tab" id="tab-${view}" data-view="${view}" aria-selected="${i === 0}" tabindex="${i === 0 ? '0' : '-1'}" aria-controls="panel-${view}-en">${bilingual(en, zh)}</button>`).join('\n')}
</div>${panels}</section>
<article class="reading-article" lang="en" data-note-language="en">${english.html}</article>
<article class="reading-article" lang="zh-CN" data-note-language="zh" hidden>${chinese.html}</article>
<footer class="reading-footer">${bilingual('Fudan University · Fall 2026 · Source evidence checked September 12, 2026.', '复旦大学 · 2026 年秋季 · 公开证据核验日期：2026 年 9 月 12 日。')}</footer>
</main></div>
<noscript><p>JavaScript enables the language switch and diagram tabs. Read the Chinese source at <a href="${slug}.zh.md">中文 Markdown</a>.</p></noscript>
</body></html>
`;
await writeFile(path.join(root, `docs/${slug}.html`), html);
console.log(`Built docs/${slug}.html and six Mermaid SVGs from bilingual source files.`);
