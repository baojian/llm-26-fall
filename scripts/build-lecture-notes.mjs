/* Render lecture notes and the course plan with local KaTeX and section links. */
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(root, 'slides/package.json'));
const MarkdownIt = require('markdown-it');
const katex = require('katex');
const escape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');
const pages = [
  { slug: 'lecture-08-instructgpt', number: '08' },
  { slug: 'lecture-09-policy-optimization', number: '09' },
  { slug: 'lecture-10-data-evaluation', number: '10' },
  { slug: 'course-revision', number: null },
];
for (const { slug, number } of pages) {
  const source = await readFile(path.join(root, 'docs', `${slug}.md`), 'utf8');
  const title = source.split('\n')[0].replace(/^# /, '');
  const math = [];
  const protectedSource = source.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+)\$/g, (_, display, inline) => {
    const index = math.length;
    math.push(katex.renderToString((display ?? inline).trim(), { displayMode: display !== undefined, throwOnError: true, strict: 'error' }));
    return `MATHPLACEHOLDER${index}END`;
  });
  const md = new MarkdownIt({ html: false });
  const headings = [];
  const tokens = md.parse(protectedSource, {});
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i].type !== 'heading_open' || tokens[i].tag !== 'h2') continue;
    const label = tokens[i + 1].content;
    const id = label.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim().replace(/\s+/g, '-');
    tokens[i].attrSet('id', id);
    headings.push(`<li><a href="#${id}">${escape(label)}</a></li>`);
  }
  const html = md.renderer.render(tokens, md.options, {}).replace(/MATHPLACEHOLDER(\d+)END/g, (_, i) => math[Number(i)]);
  const slidesLink = number ? `<a href="../slides/lecture-${number}/" data-en="Slides" data-zh="课件">Slides</a>` : '';
  const availability = number
    ? 'These lecture notes are in English. Navigation is available in English and Chinese.'
    : 'This preparation plan is in English. Navigation is available in English and Chinese.';
  await writeFile(path.join(root, 'docs', `${slug}.html`), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(title)} · Fudan LLMs</title>
<link rel="stylesheet" href="../slides/vendor/katex/katex.min.css">
<link rel="stylesheet" href="../assets/course-language.css">
<link rel="stylesheet" href="../assets/lecture-notes.css"></head>
<body><header><a href="../index.html" data-en="Course" data-zh="课程主页">Course</a>${slidesLink}<a href="${slug}.md" data-en="Markdown source" data-zh="Markdown 源文件">Markdown source</a><button onclick="window.print()" data-en="Print" data-zh="打印">Print</button><div class="lang-toggle" role="group" aria-label="Language / 语言"><button type="button" data-lang="en" lang="en" aria-pressed="true">EN</button><button type="button" data-lang="zh" lang="zh-CN" aria-pressed="false">中文</button></div></header>
<main><p class="language-availability" data-en="${availability}" data-zh="${number ? '讲义' : '备课计划'}正文目前为英文；导航支持中英文切换。">${availability}</p><details class="contents"><summary data-en="Contents" data-zh="目录">Contents</summary><ol>${headings.join('')}</ol></details><article lang="en">${html}</article></main>
<footer>CS40008.01 · Fudan University · Fall 2026</footer><script src="../assets/course-language.js"></script></body></html>\n`);
  console.log(`${slug}.html: ${headings.length} sections, ${math.length} equations`);
}
