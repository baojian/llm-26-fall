# 第 1 讲：分词之前的文本预处理

**课程：** 复旦大学，2026 年秋季，NLP 与大语言模型  
**公开资料与分词器文件核验日期：** 2026 年 9 月 12 日  
**范围：** 为分词器学习准备文本，以及在语言模型训练和推理时使用分词器。

理解这个主题，首先要区分**筛选和整理语料**与**在分词器内部变换字符串**。
生产数据流程可能包含文章提取、损坏文档过滤、代码仓库去重和语言比例调整。
调用 `tokenizer.encode(text)` 不会重新执行这些全部工作；它执行的是分词器
配置好的文本变换与已经学到的切分规则。

本讲义以 Kimi K3、GLM-5.2 和 DeepSeek-V4 为具体案例。公开文件对编码行为
提供的证据，通常多于对分词器训练样本的披露。语言模型的预训练报告，
**不能直接当作其分词器的训练配方**。

术语说明：**文本预处理（text preprocessing）**是本讲义的广义主题；
**预分词（pre-tokenization）**特指在已学习的子词模型执行之前，确定初始片段边界的步骤。

## 目录

1. [三条流程，三种任务](#1-三条流程三种任务)
2. [这些模型究竟公开了什么](#2-这些模型究竟公开了什么)
3. [实用的语料准备流程](#3-实用的语料准备流程)
4. [分词器内部的预处理](#4-分词器内部的预处理)
5. [编码时会重复哪些步骤](#5-编码时会重复哪些步骤)
6. [实例与常见错误](#6-实例与常见错误)
7. [动手检查与分词器训练](#7-动手检查与分词器训练)
8. [验证与课堂讨论](#8-验证与课堂讨论)
9. [资料阅读指南](#9-资料阅读指南)

## 1. 三条流程，三种任务

### 1.1 学习分词器

分词器学习器读取字符串或由字符串统计得到的信息，学习词表与切分规则。
普通 BPE 训练会统计相邻符号对并学习合并规则。这个过程不训练语言模型的神经网络权重。

```text
原始来源
    → 提取、修复、筛选、去重、选择
    → 具有代表性的分词器训练样本
    → 配置好的归一化与预分词
    → 学习词表和切分规则
    → 冻结完整分词器并记录版本
```

代表性样本可以远小于最终的 LLM 训练语料。样本大小、来源比例和重复内容的
处理方式都是设计选择。不要把报告中以万亿 token 计的语言模型训练预算，
误认为学习分词器所用的文本量。

### 1.2 训练语言模型

分词器固定后，用它编码选定的语言模型语料。训练样本构建还要处理文档边界、
数据格式、序列拼接（packing）、注意力掩码与损失掩码。有些格式化在编码前完成，
有些操作直接作用于编码后的 ID。

```text
准备好的语言模型语料
    → 按需构建文本或消息格式
    → 使用冻结的分词器编码
    → 构建训练序列与掩码
    → 训练语言模型
```

分词器训练样本与 LLM 训练语料需要兼容的文本约定，但不必使用完全相同的抽样比例。

### 1.3 运行语言模型

```text
用户文本、消息、文件或检索文档
    → 应用所需的内容提取与消息序列化
    → 使用匹配的冻结分词器编码
    → 模型推理
```

对于普通字符串，可能无需内容提取。PDF 附件需要文档处理器或视觉输入路径；
对话消息需要遵循模型的消息格式。这些操作都不能简单地等同于词表查找。

分词器本身通常包括归一化、预分词、已学习的切分模型和后处理。
相关库将它们作为独立组件暴露出来。[Hugging Face 分词流程][hf-pipeline]

## 2. 这些模型究竟公开了什么

### 2.1 如何理解证据

本讲义区分三类证据：

| 类别 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| **报告披露** | 模型开发者说明了某项训练数据实践。 | 全部实现细节、阈值或分词器训练样本。 |
| **文件与实测核验** | 特定公开文件包含某条规则，或本地检查观察到某种行为。 | 未公开的上游清洗流程或托管 API 的行为。 |
| **实践建议** | 为课程项目或生产流程提出的可操作设计。 | 某家公司确实使用这一实现。 |

下面固定了本次核验的版本。以后访问 `main` 可能得到不同文件，因此复现实验时应使用固定版本链接。

| 发布版本 | 核验的仓库 revision | 编码证据 |
| --- | --- | --- |
| `moonshotai/Kimi-K3` | `f831ab66814297da540d832a5235f8e904f29d06` | [分词器实现][kimi-tokenizer]、[BPE ranks][kimi-ranks]、[配置][kimi-config] |
| `zai-org/GLM-5.2` | `cf457fa734ab149ffef225f80893eb38c6ff5cdc` | [分词器 JSON][glm-tokenizer]、[配置][glm-config]、[对话模板][glm-template] |
| `deepseek-ai/DeepSeek-V4-Pro` | `b5968e9190ef611bbf34a7229255be88a0e937c1` | [分词器 JSON][ds-tokenizer]、[配置][ds-config]、[消息编码说明][ds-encoding] |

### 2.2 Kimi K3

**报告披露的语料准备。** 第 3.1 节介绍了网页、代码、数学、知识类文本，以及视觉语料。
文本筛选结合启发式规则、质量分类器和去重；领域抽样率通过小模型消融实验确定。
知识与数学语料还进行了受控改写，采用多样化提示、按块生成和与原文的保真核验。
第 3.4 节描述了长输入的额外清洗，包括精确与模糊去重、质量筛选和结构验证。
这些属于训练数据处理，不是 `encode()` 对每条用户提示执行的步骤。
[Kimi K3 报告，第 3.1、3.4 节][kimi-report]

**文件核验的编码行为。** 公开实现使用 tiktoken BPE。正则表达式将汉字连续片段
单独划分，使用区分大小写的分支处理非汉字字母，以 `\p{N}{1,3}` 对数字分组，
并以专门分支保留空白。所检查的普通文本路径没有转小写或 Unicode 归一化调用。
对话路径会区分结构标记和普通消息内容，再决定特殊 token 的字面拼写是否映射为特殊 ID。
非常长的字符串还会经过实现规定的大小限制处理。[Kimi K3 分词器实现][kimi-tokenizer]

**这些资料尚不能确定：** 精确的分词器训练样本、语言比例、全部上游文本修复，
以及可完全复现的语料清洗配置。报告中名义上的 160K 词表规模，本身不能证明
K3 与较早 Kimi 版本使用相同的 token ID 或预处理。

### 2.3 GLM-5.2

**本版本的证据。** GLM-5.2 发布了自己的分词器文件与对话模板，本讲义据此说明
其编码行为。官方 GLM 仓库链接的是 GLM-5 技术报告；核验过的 GLM-5.2 发布材料
没有给出一份完整、独立可复现的分词器训练或语料预处理配方。
[GLM 官方仓库][glm-repo]、[GLM-5.2 模型卡][glm-card]

**较早同系列的证据，明确属于 GLM-5。** 报告第 2.2 节介绍了额外的 DCLM 风格
分类器，以及用于网页筛选的世界知识分类器。代码准备包括模糊去重、修复
Software Heritage 元数据对齐问题、改进语言分类，以及按质量抽样。
数学与科学数据改进了网页提取和 PDF 解析，通过教育价值评分筛选内容，
并聚合分块评分来评估长文档。这一数学/科学流程还过滤合成和模板化材料。
这些是具体的实践案例，但不能据此断言 GLM-5.2 沿用了完全相同的规则或阈值。
[GLM-5 报告，第 2.2 节][glm-report]

**文件核验的 GLM-5.2 编码。** `normalizer` 为 `null`。先执行正则切分，再进行
`ByteLevel` 转换，其中 `add_prefix_space=false`、`use_regex=false`；后者避免
再次执行默认正则表达式。数字按 1–3 个字符分组；通用字母分支可以同时包含拉丁
字母与汉字。包装配置还设置了 `do_lower_case=false`、`remove_space=false`
和 `clean_up_tokenization_spaces=false`。这些设置描述的是所检查实现的编码与
解码行为，而非最初的网页提取器。[分词器 JSON][glm-tokenizer]、[配置][glm-config]

### 2.4 DeepSeek-V4

**报告披露的语料准备。** 第 4.1 节介绍了过滤批量自动生成和模板化网页、扩大
多语言覆盖，以及重点整理长篇科学与技术文档。数学和代码仍是核心内容。
报告说明预处理大体沿用 V3，并在 V3 分词器基础上添加上下文构建用的特殊 token，
同时保持名义上的 128K 词表。V4 还沿用 token 拆分和中间填空（FIM）训练，
通过文档拼接减少截断，并使用样本级注意力掩码。后面这些操作属于训练样本构建，
而非普通字符串清洗。[DeepSeek-V4 报告，第 4.1 节][ds-report]

**文件核验的 V4-Pro 编码。** 归一化器是空序列，不执行文本归一化。
预分词依次进行：隔离 1–3 位数字组；隔离指定中文/日文字符范围的连续片段；
应用进一步的字母、标点和空白规则；最后做 `ByteLevel` 转换，不额外添加前导
空格，也不再执行第二套正则。这里指定的字符范围不等于全部 Unicode CJK 字符。
[DeepSeek-V4-Pro 分词器 JSON][ds-tokenizer]

**值得理解的训练期例外。** V3 报告描述了在训练时随机拆分部分同时包含标点和
换行的 token，以减少 token 边界偏差。V4 明确沿用 token 拆分。
这样会让 LLM 在训练中看到更多切分形式，但并不意味着推理时应随机修改标点，
或重新运行 BPE 学习算法。[DeepSeek-V3 报告，第 4.1 节][ds-v3-report]、
[DeepSeek-V4 报告，第 4.1 节][ds-report]

**尚不能确定：** 精确过滤阈值、完整的分词器训练清单，以及全部 V4 变体和
托管服务是否具有完全一致的行为。后面的文件实测只对应固定 revision 的 V4-Pro。

### 2.5 已发布文本编码路径的直接比较

| 属性 | Kimi K3 | GLM-5.2 | DeepSeek-V4-Pro |
| --- | --- | --- | --- |
| 已学习的编码方式 | tiktoken BPE | BPE | BPE |
| 普通文本归一化的观察结果 | 所检查路径无归一化调用 | `null` 归一化器 | 空归一化序列 |
| 数字预分词 | 每组 1–3 位 | 每组 1–3 位 | 每组 1–3 位 |
| 汉字处理 | 明确的汉字连续片段分支 | 通用字母分支可跨文字系统 | 对指定中文/日文范围单独切分 |
| 空白处理 | 专门的正则分支 | 正则切分后转为字节表示 | 多次切分后转为字节表示 |
| 消息格式 | 自定义分段对话编码 | 已发布的 Jinja 模板 | 已发布的消息编码器 |

来源：[Kimi 实现][kimi-tokenizer]、[GLM 分词器][glm-tokenizer]、
[GLM 模板][glm-template]、[DeepSeek 分词器][ds-tokenizer]、
[DeepSeek 消息编码器][ds-encoding]。

它们都属于相近的字节子词编码方案，但**不能用一个模型的正则、词表或消息模板
来替代另一个模型的对应组件**。

## 3. 实用的语料准备流程

下面的流程是**建议的设计**，并非对某家实验室私有流程的重建。
实际系统会按来源类型分支，也可能多轮执行筛选或去重。廉价过滤常在昂贵的模型
评分之前进行；后续质量评估也可能决定重复簇中保留哪个文档。

DataTrove 提供 reader、extractor、filter、deduplication 和 writer 等组件，
并包含 FineWeb 流程示例。Dolma 也将原始文档与派生标注分开存放。
这些可运行的工业实践，比一个通用 `clean_text()` 函数更适合作为实现参考。
[DataTrove][datatrove]、[Dolma 数据格式][dolma-format]

### 3.1 定义目标数据并保留来源信息

处理前，先确定模型必须支持哪些语言、领域和格式。编程助手、多语言聊天模型
与科学模型，不应默认采用完全相同的语料比例。

每条文档记录至少保留：

- 稳定的 ID，以及来源地址或来源 ID。
- 采集时间和来源快照/版本。
- 声明或检测到的内容类型与字符编码。
- 代码仓库路径、文档标题等来源特定元数据。
- 适用的来源使用信息与采集排除条件。
- 提取/清洗版本、质量得分与拒收原因。

除非训练格式明确要求，否则将元数据与模型可见文本分开。否则 JSON 键、
抓取时间戳和存储路径可能意外影响分词器词表。

一个简单记录可以采用以下结构：

```json
{
  "id": "example-document-001",
  "source_type": "html_article",
  "text": "A paragraph extracted from an article.\n",
  "metadata": {
    "snapshot": "course-demo-2026-09",
    "language": "en",
    "extractor_version": "article-extractor-v1",
    "decision": "keep"
  }
}
```

第 7 节的简单训练示例只把选中的 `text` 字段送入学习器。
更复杂的格式应明确列出额外字段。

### 3.2 正确解码来源字节

文件以字节形式存储，多数分词器 API 接收 Unicode 字符串。

1. 根据可靠的来源元数据确定格式与预期字符编码。
2. 采用明确的解码策略，并检测格式错误的输入。
3. 记录证据，对失败样本进行隔离或修复。
4. 用一致的编码序列化准备好的语料，通常是 UTF-8。

需要发现的问题包括：把旧字符编码误当 UTF-8、文本被重复解码、HTML 实体仍然
以字面形式存在，以及二进制载荷被误当作自然语言文本。

不要将 `errors="ignore"` 当作通用修复方式，因为它会静默删除字节。
替换式解码也可能插入 `U+FFFD` 字符。应统计这些替换，并保留原始来源以便诊断。

**文件解码与字节级分词不是同一步。** 先把文件正确解码为文本；随后字节级
分词器再通过字节表示该文本，通常使用 UTF-8 字节。字节级 BPE 无法恢复文件
错误解码时已经丢失的信息。

### 3.3 按格式提取内容

| 来源 | 有用的提取行为 | 典型失败 |
| --- | --- | --- |
| HTML 文章 | 解析 DOM、识别正文、保留段落与列表、只解码一次实体 | 导航和 cookie 横幅淹没正文 |
| 含数学内容的 HTML | 提取正文时保留 LaTeX 标注等公式来源 | 删除标签时公式一起消失 |
| PDF 或扫描页 | 恢复阅读顺序、表格、公式和图注；按需 OCR | 多栏内容交错，页眉反复出现在句子中间 |
| 源代码仓库 | 识别文本文件与编程语言；保留文件边界、缩进和相关路径 | 生成文件占据主要比例，制表符或换行被破坏 |
| Notebook | 显式解析单元格并决定保留哪些输出 | 图片、报错堆栈或执行元数据变成训练文本 |
| JSON 或工具轨迹 | 解析 schema、选择目标字段、保留消息与工具间关系 | 将传输协议外壳误认为对话内容 |

OpenWebMath 说明了通用网页提取为何不足：忠实保留数学表达是其数据集构建的
核心要求。即使公式的 HTML 表示不是可见正文，它依然属于文档内容。
[OpenWebMath 论文][openwebmath]

并非所有任务都应该去掉 HTML。训练模型编写或理解网页源码时，需要 HTML
样本。应根据预期用途分流文档，而不是删除所有 `<...>`。

### 3.4 保守地修复提取错误

一些依赖来源类型的有用修复包括：

- 去掉 PDF 提取器引入的重复页眉。
- 在原始版面提供依据时，重新连接被换行断开的单词。
- 统一提取器生成的段落分隔符。
- 在 HTML 提取过程中解码 HTML 实体。
- 排除包含截断二进制数据的损坏记录。

每次修复都在改变文本分布。要区分提取错误与作者原本希望保留的内容。
例如 `inter-\nnational` 可能是 PDF 排版断词，而代码中的减号加换行可能有意义。
不要对两者都套用全局连字符删除规则。

对会被重复应用的确定性修复，可以检查幂等性：
`repair(repair(text)) == repair(text)`。幂等性有帮助，但不证明语义正确。
例如重复进行 HTML 反转义，会改变那些本来就要展示转义实体的示例。

### 3.5 识别语言、领域与文档结构

语言识别可以用于决定收录、分流和抽样。领域标签有助于区分自然语言、代码、
数学、参考材料和对话。长文档可能需要分块评分，因为开头的一小段不一定代表全文。

不要直接推广只适用于英语的启发式规则。要求至少包含若干个空格分隔单词的
规则，在中文上的行为不同。代码片段和混合语言网页，也可能得到不可靠的
文档级语言标签。

对于多语言课程语料，应保留标签置信度，人工检查低置信度及混合语言样本。
语言标签应当是可核验的判断，而不是把全部输入翻译成英语的理由。

### 3.6 综合多种证据筛选质量

先用廉价信号识别明显问题，再在确有收益的位置使用更昂贵的评分。

| 信号 | 可能发现的问题 | 过滤前需要检查的内容 |
| --- | --- | --- |
| 提取结果为空或异常短 | 提取失败或只有导航的页面 | 合法的短定义、对话和代码 |
| 重复行或 n-gram | 重复模板、循环生成文本 | 表格、诗歌、列表与重复性算法 |
| 行异常长 | 压缩前端资源、编码载荷 | 合法 JSON、公式和有用的生成源码 |
| 字符分布异常 | 损坏文本或类似二进制的数据 | 非拉丁文字、emoji 和数学记号 |
| 语言模型困惑度 | 不符合参考分布的文本 | 专业术语与低资源语言 |
| 学习得到的质量/教育价值评分 | 简单规则难以区分的内容 | 分类器偏差与分布外失败 |

保留决策阈值附近的样本，并抽查被排除的文档。按语言和领域分别统计保留率。
提升平均英语正文质量的过滤器，仍可能大量删除代码或中文。

FineWeb 是一个公开案例，结合了内容提取、语言选择、质量筛选和 MinHash
去重。其配方针对特定网页语料，而非对“好文本”的通用定义。
[FineWeb 数据集卡][fineweb]

### 3.7 在合适的粒度去重

至少区分三种情况：

1. **精确重复：** 文档内容完全相同，通常可用哈希识别。
2. **近似重复：** 文档大部分重叠，仅有少量编辑、页眉或镜像差异；基于
   shingle 的 MinHash 是一种选择。
3. **重复片段：** 其他内容不同的文档中，共享某些段落或模板文本。

近似去重需要相似性定义、候选检索、阈值和保留策略。它不等于语义等价判断。
对同一个定理的两种解释，可能都有价值，也不一定属于文本重复。

如果匹配需要较强的规范化，应保存独立的**比较表示**。去重键转小写，不要求
送入模型的文本也转小写。对代码去重时，应避免使用会抹掉缩进或标识符大小写的表示。

重复内容会改变符号对频率，因此去重会影响分词器学习。但删除每个重复词或
常见句子，也会破坏有用的频率信息。应以适合数据的文档/片段粒度去重，再显式
选择抽样权重。Dolma 记录了基于文档字段和段落的可配置去重。
[Dolma 去重文档][dolma-dedup]

### 3.8 应用内容排除策略并隔离评测数据

对于生产数据集，需要明确处理不希望保留的个人联系信息、意外进入代码的密钥，
以及其他按来源排除的内容。根据记录及用途，可以删除文档、掩盖片段，或保留
合法的公开示例。正则匹配本身不能完成对内容的全部判断。

Dolma 披露了对部分个人信息类别的掩盖，说明这些决策可以属于语料准备。
这种掩盖不是 BPE 的固有属性。[Dolma 论文][dolma-paper]

评测材料应与训练数据隔离。通过有记录的比较表示搜索精确及近似重叠。
必要时将相关文档或仓库作为一组划分，避免近重复样本同时出现在训练集和验证集。

分词器实验也应把评估文本排除在词表学习之外。在分词器训练中见过评测文本，
不同于训练 LLM 时见过基准答案，但仍会影响对压缩效率或覆盖率进行独立留出评估。

### 3.9 选择分词器训练配比

这一步容易被忽视。在预分词边界和学习算法约束下，BPE 按观察到的统计信息
分配词表容量。因此，大规模英语主导样本可能导致某个规模较小但重要的领域压缩较差。

一种实用做法是：

1. 将整理后的语料按语言和领域分层。
2. 在每一层选择可复现的样本。
3. 限制极端文档长度，或分块抽样，同时保留真实空白和文档结构。
4. 使用明确的混合权重训练候选分词器。
5. 按分层比较留出文本的 token 数和保真度。
6. 根据结果调整配比和词表预算。

必须说明权重的单位：文档数、字符数、UTF-8 字节数，或某个具名参考分词器下的
token 数。文档数相同，不代表文本量相同；字节数相同，也不代表语言覆盖相同。

如果没有公开依据，不要声称某个分词器使用了例如 50/30/20 的训练配比。
本讲义没有给这三个模型断言任何精确的分词器训练比例。

### 3.10 固定文本约定并输出可复现的数据分片

记录内容提取规则、可选归一化、预分词器、特殊 token 策略、样本清单、随机种子
和库版本。采用稳定的顺序或有记录的随机打乱。保留记录边界，并以可恢复处理的
分片存储输出，同时保存计数和校验和。

优先逐条流式读取，而非将整个语料拼为一个字符串。对于 JSONL，应解析每条
JSON 记录并产出真正的 `text` 字段，不要意外学习 JSON 序列化语法。
JSON 文件中的字面量 `\n` 在解析后会变成换行。

学习器仍需内存保存统计信息，因此流式迭代器不意味着分词器训练的内存是常数。
扩大规模前，应测量不同预分词片段的数量，并检查异常长片段。

### 3.11 一份有公开参数的配方：FineWeb

作为与部分披露的模型流程的对照，FineWeb 数据集卡给出了以下步骤：

1. 按来源排除条件过滤 URL。
2. 使用 Trafilatura 从 Common Crawl WARC 中的 HTML 提取文本。
3. 保留 fastText 英语得分至少为 **0.65** 的文档。
4. 使用 Gopher、部分 C4 和额外的 FineWeb 质量规则；不采用 C4 的句末标点要求。
5. 对每次 crawl 单独做 MinHash 去重，使用 **5-gram** 和 **14 × 8** 个哈希函数。
6. 掩盖邮件地址和公网 IP 地址。

这是一份带有可检查参数的具体语料配方，但不能据此断言 Kimi、GLM 或 DeepSeek
采用相同参数。若目标语料要保留中文，其中的英语筛选条件就不适用。
[FineWeb 数据集卡，数据处理部分][fineweb]

## 4. 分词器内部的预处理

### 4.1 归一化是一项设计选择

Unicode 归一化让某些 Unicode 表示形式一致。它不同于转小写、去重音、语言转换
或空白删除。

| 选择 | 例子 | 需要考虑的信息 |
| --- | --- | --- |
| 不做归一化 | 将 `café` 与 `cafe\u0301` 保留为不同码点序列 | 视觉相同的词可能具有不同切分 |
| NFC | 在适用时把 `e` 加组合重音合成为 `é` | 改变码点表示，但保留规范等价关系 |
| NFKC | 将兼容形式 `Ａ` 变成 `A`、`①` 变成 `1` | 可能抹掉记号或精确文本复现所需的区别 |
| 转小写 | `US` → `us` | 丢失大小写，包括名称和代码中的区别 |
| 折叠空白 | 四个空格 → 一个空格 | 可能破坏缩进、对齐和格式 |

NFC 与 NFKC 的等价目标不同；两者都不意味着“删除一切不常见字符”。
[Unicode 归一化规范][unicode]

训练新的通用或代码分词器时，可以先以保留输入为基线进行评估。
使用已有模型时，则遵循其发布配置。第 2 节核验的模型不构成额外添加 NFKC
或转小写步骤的依据。

SentencePiece 提供了有用对照：它通常采用基于 NFKC 的归一化规则，支持自定义
规则，并将规则保存进模型。`identity` 关闭这类 Unicode 改写，但仍需检查
空白相关选项。“使用 SentencePiece”也没有说明所学模型究竟是 BPE 还是 Unigram。
[SentencePiece 归一化文档][sentencepiece-normalization]、[选项说明][sentencepiece-options]

### 4.2 预分词确定候选边界

预分词器将文本分成若干片段，供后续子词模型处理。在这里研究的 BPE 流程中，
合并不能跨越这些预分词边界，但一个片段仍可能产生多个最终 token。
[Hugging Face 分词流程][hf-pipeline]

例如 `\p{N}{1,3}` 将数字连续片段切为至多三个 Unicode 数值字符一组。
它不保证每组对应一个最终 token，也不表示“完整的数”。小数点、符号和分隔符
还会受到其他规则处理。

以下结果来自固定版本文件的实测：

| 输入 | Kimi K3 预分词片段 | GLM-5.2 预分词片段 | DeepSeek-V4-Pro 预分词片段 |
| --- | --- | --- | --- |
| `2026 1234567` | `['202', '6', ' ', '123', '456', '7']` | 相同 | 相同 |
| `Hello世界ABC` | `['Hello', '世界', 'ABC']` | `['Hello世界ABC']` | `['Hello', '世界', 'ABC']` |
| `Hi.\nNext` | `['Hi', '.\n', 'Next']` | 相同 | 相同 |

第二个例子不代表 GLM 为整个字符串只输出一个 token；最终实测得到三个 ID。
第一个例子中，Kimi 和 DeepSeek 输出六个 ID，GLM 则输出七个，因为其中的
`'456'` 片段最终拆成 `'45'` 与 `'6'`。预分词表与最终 token 表回答的是不同问题。
[Kimi ranks][kimi-ranks]、[GLM 分词器][glm-tokenizer]、[DeepSeek 分词器][ds-tokenizer]

Python 内置 `re` 不支持这些模式使用的 Unicode 属性语法。后面的 Kimi 检查
使用 `regex` 包及 version-1 集合语义；GLM 和 DeepSeek 则直接使用 JSON 中
保存的预分词器。不要静默改用近似正则后，声称精确复现了原模型。

这些公开流程不要求先用外部分词器把中文切成词。额外插入词间空格会改变输入
分布。token 边界可能位于单词、中文表达甚至 emoji 序列内部；它不是语言学标注。

### 4.3 字节转换保证可表示性，不保证语义完整

在字节 BPE 中，一个字符可以跨多个字节，一个 token 也可以覆盖部分字符或
多个字符。某些实现内部使用的字节到符号映射，是可逆表示，不是 Unicode 清洗。

覆盖全部 256 种字节值的字节级初始字母表，可以编码有效输入文本，不需要为每个
词预先准备完整词条。但这不意味着所有文字系统的压缩效率相同：罕见字符仍可能
消耗多个 ID。不能只凭是否出现未知 token 来评估多语言质量。

JSON 中的 `byte_fallback=false`，不代表该分词器缺乏字节覆盖。
字节级字母表与为其他基础字母表提供后备编码的机制，是两种不同设计。

### 4.4 学习并冻结分词器

明确词表预算、初始字母表、特殊 token 和训练参数。对 BPE，有用的控制项包括
最小符号对频率和最大学习 token 长度；否则高度重复的材料可能让词表容量花在
很长的重复字符串上。[Hugging Face trainer API][hf-trainers]

只保存词表不够。应按实现保存合并规则或得分、归一化器、预分词器、解码器、
added-token 定义、包装配置和消息格式实现，并将其版本与模型 checkpoint 一同记录。

还要区分词表计数口径：

- 核验的 Kimi rank 文件有 **163,584** 个可合并词条，包装层另外预留 **256** 个 ID。
- 核验的 GLM JSON 有 **154,820** 个模型词表条目；计入 added tokens 后共有
  **154,856** 个不同 ID；模型配置包含 **154,880** 个词表槽位。
- 核验的 DeepSeek JSON 有 **128,000** 个模型词表条目；计入 added tokens 后
  共有 **129,280** 个不同 ID。部分 added-token 记录与已有 ID 重叠，因此直接
  相加记录数会出错。

这些是文件中的计数，不是对语料规模的另一种估计。
[Kimi ranks][kimi-ranks]、[Kimi 实现][kimi-tokenizer]、[GLM 分词器][glm-tokenizer]、
[GLM 模型配置][glm-model-config]、[DeepSeek 分词器][ds-tokenizer]

## 5. 编码时会重复哪些步骤

### 5.1 实际使用中的回答

| 操作 | 学习分词器之前 | 训练 LLM 时 | 普通推理时 |
| --- | --- | --- | --- |
| HTML/PDF 提取 | 来源需要时执行 | 来源需要时执行 | 应用接收这类来源时执行 |
| 语料质量筛选 | 通常属于样本选择 | 属于语料选择 | 不是分词器执行的操作 |
| 语料去重和配比抽样 | 训练样本的设计选择 | LLM 语料的设计选择 | 分词器不会对每条提示执行 |
| 必要的确定性文本准备 | 遵循选定的输入约定 | 使用兼容约定 | 如果属于应用输入约定则执行 |
| 分词器归一化 | 按配置执行 | 按配置执行 | 按配置执行 |
| 分词器预分词 | 按配置执行 | 按配置执行 | 按配置执行 |
| 学习 BPE 合并或 Unigram 参数 | 是 | 通常否，分词器已冻结 | 否 |
| 对话/工具序列化 | 仅当刻意纳入训练样本 | 对应训练样本需要 | 对应模型接口需要 |
| 训练增强或替代切分 | 仅当属于分词器实验 | 可能使用 | 通常关闭，除非明确启用 |
| 拼接、填充、注意力掩码、损失掩码 | 不属于词表学习清洗 | 为训练构建 | 可能有批处理/上下文处理；没有训练损失掩码 |

所以，“相同预处理”意味着**兼容的模型可见文本，以及匹配的分词器规则**，
不是对每条消息重新运行整个数据工厂。

外部预处理不会自动保存到分词器里。若在独立脚本中把训练语料转小写，却没有
配置 lowercase normalizer，保存后的分词器不会自动对未来输入转小写。
应把必要变换放入支持的分词器组件，或作为应用流程的一部分单独维护并记录版本。

### 5.2 序列化属于模型接口

消息列表还不是模型最终的输入序列。角色、轮次边界、工具 schema 和生成前缀
需要按模型预期表示。即使各条消息的内容字符串未改动，应用对话模板也会改变输入。

对于受支持的 Hugging Face 对话模板，直接调用
`apply_chat_template(..., tokenize=True)` 可以避免常见的特殊 token 重复插入。
若先渲染完整模板为字符串，再单独编码，通常应使用 `add_special_tokens=False`，
避免再次添加边界 token。自定义编码器替代通用路径时，以模型专用说明为准。
[Hugging Face 对话模板][hf-chat]

例如，DeepSeek 发布的消息编码器处理对话、推理与工具调用结构。它构造提示词，
不是网页语料清洗器。GLM 提供 Jinja 模板，Kimi 则使用自定义分段消息路径。
[DeepSeek 编码说明][ds-encoding]、[GLM 模板][glm-template]、[Kimi 分词器][kimi-tokenizer]

特殊 token 匹配也是简单四框流程图不够完整的原因。相关库可能在普通 BPE 路径
之外识别 added tokens。消息内容中的字面标记，必须遵循模型的序列化器和 token
匹配策略处理。任意拼接文本，不能保证复现官方消息路径。

### 5.3 确定性编码存在明确例外

固定普通文本、分词器文件、运行库和选项后，常规推理编码是确定的。
需要区分显式启用的采样或 BPE dropout、仅训练期使用的替代切分，以及实现对
非常长字符串的特定处理。

即使没有随机性，分别编码片段也可能改变 ID：

```text
encode("hello world")
可能不同于
encode("hel") + encode("lo world")
```

是否不同取决于切分边界和词表。每隔 N 个字符随意切开长文件，可能保留解码后的
文本，却改变编码切分。匹配已有模型时，应采用其官方运行实现的行为。

## 6. 实例与常见错误

### 6.1 含公式的网页文章

假设一个 HTML 页面包含导航、正文、cookie 横幅，以及存放在数学标注中的公式。

```text
源 HTML
    → 解析并选择文章正文
    → 将公式保留为明确的文本表示
    → 保留段落边界
    → 质量评估与去重
    → 按需纳入分词器训练样本
```

之后，若用户将已提取正文输入聊天框，分词器不需要再运行 HTML 提取器。
如果检索系统抓取的是 HTML 页面，检索系统就需要兼容的提取路径。

不要为了让文本“更干净”而删除全部标点：`x != y`、`x = y`、`x < y`
表达的是不同命题。

### 6.2 一个 Python 函数

```python
def sign(x):
    if x > 0:
        return 1
    return -1
```

根据选定的来源策略，保留缩进、标识符、运算符、注释和行结束符。
不要把标识符转小写或删除标点。决定是否排除自动生成的依赖目录，属于语料选择；
保留已收录文件的缩进，则属于文本保真要求。

对于代码中间填空，训练流程可能用特殊标记把文档重排成 prefix/suffix/middle。
这是训练任务格式，不意味着普通编码会重排每段代码。

### 6.3 两个看起来相似的 Unicode 字符串

`"café"` 与 `"cafe\u0301"` 看起来可能一样，却具有不同的码点序列。
保留原输入的分词器可能输出不同 ID。NFC 归一化可以在切分前使它们一致。
这两种行为本身都不一定是错误。

检查过的分词器路径在测试样本上保留了两种形式。额外调用
`unicodedata.normalize("NFKC", text)` 或 `text.lower()` 会加入新的应用级变换，
而这些公开文件并未要求这样做。

### 6.4 在标点处结束的不完整提示词

`"Answer:\n"` 与 `"Answer:"` 的 token 序列未必互为前缀，因为已学习的一个
token 可能同时包含冒号和换行。这会影响 few-shot 格式比较和前缀缓存。
这类边界敏感性与第 2.4 节的 DeepSeek-V3 训练干预有关，但不构成删除换行的理由。
[DeepSeek-V3 报告][ds-v3-report]

### 6.5 不宜作为通用默认值的变换

| 看似方便的默认操作 | 可能的损害 | 更合适的决策 |
| --- | --- | --- |
| 删除停用词 | 丢失否定、关系和正常语法 | 通用 LLM 训练保留自然文本 |
| 全部词干化或词形还原 | 改变模型需要复现的文本 | 只用于明确针对该表示定义的任务 |
| 全部转小写 | 丢失大小写语义与代码标识符 | 遵循目标分词器；新训练时评估归一化选择 |
| 删除标点或数字 | 破坏代码、数学、日期和数值 | 保留内容，单独过滤损坏文档 |
| 折叠全部空白 | 破坏缩进、表格和有意的换行 | 按格式修复 |
| 不评估就使用 NFKC | 可能合并不同符号或文本形式 | 明确选择并测量后果 |
| 先对中文做词级切分 | 加入原模型没有的边界 | 使用模型自己的预分词器 |
| 将所有语言译成英语 | 改变语料和目标能力 | 显式选择与平衡语言 |
| 总是用 LLM 改写文本 | 可能改变事实、风格和频率 | 将改写视为需单独验证的增强 |

## 7. 动手检查与分词器训练

这些示例不需要模型权重或 GPU。将代码保存到 Git 忽略的 `workspace/` 中。
它们用于检查和教学，不能完整替代各模型的消息处理器。

### 7.1 检查真实文件，而不执行模型仓库代码

将下面的代码保存为 `workspace/inspect_tokenizer_preprocessing.py`，
在仓库根目录运行：

```bash
uv run --with tokenizers==0.23.2 --with tiktoken==0.14.0 --with regex==2026.9.3 python workspace/inspect_tokenizer_preprocessing.py
```

首次运行只下载分词器 JSON、BPE ranks 和一个用于静态检查的 Python 源文件。
源文件只会被解析为 AST，不会导入或执行。下面重建的 Kimi 编码只处理不含特殊
标记的普通文本，不实现 Kimi 对话、多模态处理或包装层的长输入限制。

```python
import ast
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import regex
import tiktoken
from tokenizers import Tokenizer


ROOT = Path("workspace/tokenizer-artifacts")
ROOT.mkdir(parents=True, exist_ok=True)
RELEASES = {
    "kimi": ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"),
    "glm": ("zai-org/GLM-5.2", "cf457fa734ab149ffef225f80893eb38c6ff5cdc"),
    "deepseek": ("deepseek-ai/DeepSeek-V4-Pro", "b5968e9190ef611bbf34a7229255be88a0e937c1"),
}
SAMPLES = [
    "2026 1234567",
    "Hello世界ABC",
    "Hello WORLD!",
    "ＡＢＣ ① café cafe\u0301",
    "if x:\n    print(x)\n",
    "Hi.\nNext",
    "👩\u200d💻",
]


def fetch(name, filename):
    repo, revision = RELEASES[name]
    path = ROOT / f"{name}-{revision}-{filename}"
    if not path.exists():
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print("artifact", path.name, "sha256", digest)
    return path


for name in ("glm", "deepseek"):
    path = fetch(name, "tokenizer.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(path))
    print(name, "normalizer", config["normalizer"])
    print(name, "pre_tokenizer", config["pre_tokenizer"])
    for text in SAMPLES:
        # ByteLevel displays byte-mapped symbols internally. Recover the
        # source spans using offsets for these particular probe strings.
        spans = tokenizer.pre_tokenizer.pre_tokenize_str(text)
        chunks = [text[start:end] for _, (start, end) in spans]
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        restored = tokenizer.decode(ids, skip_special_tokens=False)
        assert restored == text, (name, repr(text), repr(restored))
        print(name, repr(text), "chunks", chunks, "ids", ids)


source = fetch("kimi", "tokenization_kimi.py").read_text(encoding="utf-8")
tree = ast.parse(source)
assignment = next(
    node for node in ast.walk(tree)
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "pat_str"
            for target in node.targets)
)
pattern = "|".join(ast.literal_eval(assignment.value.args[0]))
splitter = regex.compile(pattern, regex.VERSION1)
rank_lines = fetch("kimi", "tiktoken.model").read_bytes().splitlines()
ranks = {
    base64.b64decode(token): int(rank)
    for token, rank in (line.split() for line in rank_lines)
}
encoding = tiktoken.Encoding(
    name="kimi-k3-ordinary-text-inspection",
    pat_str=pattern,
    mergeable_ranks=ranks,
    special_tokens={},
)
for text in SAMPLES:
    chunks = splitter.findall(text)
    assert "".join(chunks) == text
    ids = encoding.encode_ordinary(text)
    assert encoding.decode(ids) == text
    print("kimi", repr(text), "chunks", chunks, "ids", ids)
```

在核验的文件与软件包版本下，三个编码器对这七个普通文本样本均能精确编码后
还原。这个小检查不能证明任意字节串、全部 Unicode 边界情况、对话模板或托管
服务的行为。在组合重音示例中，GLM 预分词器把末尾的组合重音单独切开，而
Kimi 与 DeepSeek 将其保留在前面的字母片段中；三个编码器都保留了最终文本。

### 7.2 训练一个小型字节级示例分词器

该示例展示训练 API 中预处理放在哪里。它采用**教学配置**，不是 Kimi、GLM
或 DeepSeek 的正则或训练参数。小词表用于展示机制，不适合评价生产质量。

保存为 `workspace/train_demo_tokenizer.py`，然后运行：

```bash
uv run --with tokenizers==0.23.2 python workspace/train_demo_tokenizer.py
```

```python
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers


samples = [
    "Hello world!\n",
    "Hello世界ABC\n",
    "Numbers: 2026 and 1234567.\n",
    "def sign(x):\n    return 1 if x > 0 else -1\n",
    "ＡＢＣ ① café cafe\u0301 👩\u200d💻\n",
]
tokenizer = Tokenizer(models.BPE())
# No normalizer: retain the input code-point sequence.
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
])
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=400,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<eos>"],
    max_token_length=24,
    show_progress=False,
)
tokenizer.train_from_iterator(samples, trainer=trainer)
path = Path("workspace/demo-tokenizer.json")
path.parent.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(path))
loaded = Tokenizer.from_file(str(path))

for text in samples + ["Unseen text: naïve, 中文, 987654321.\n"]:
    before = tokenizer.encode(text, add_special_tokens=False).ids
    after = loaded.encode(text, add_special_tokens=False).ids
    assert before == after
    assert loaded.decode(after, skip_special_tokens=False) == text
    print(repr(text), after)
```

这里配置的预分词器会同时用于学习和编码。256 字节初始字母表通过编码字节来
支持未见字符。注册 `<eos>` 是预留特殊 token，不会自动给每条记录追加该 token。
指定词表大小是预算；微型语料可能没有足够的合格合并对，无法达到该大小。
[Hugging Face trainer API][hf-trainers]

如果使用准备好的 JSONL 语料，应把训练调用中的 `samples` 替换为逐条解析并
产出 `record["text"]` 的迭代器。语料选择在上游完成，不要把完整清洗流程变成
用户每提问一次就执行一次的 normalizer。

## 8. 验证与课堂讨论

### 8.1 扩大规模之前测量什么

| 检查项 | 意义 |
| --- | --- |
| 提取前后的样例 | 发现丢失的公式、表格、代码与阅读顺序 |
| 按来源、语言和领域统计保留文档/字节数 | 发现非预期的配比变化 |
| 拒收原因及被拒收样本 | 使质量筛选可检查 |
| 重复簇大小及保留策略 | 发现过度匹配和占比过高的镜像 |
| 预处理后的训练/验证重叠 | 判断留出比较是否有意义 |
| 各语言、领域的 token 数 | 发现被总体平均值掩盖的压缩问题 |
| 编码解码后的文本比较 | 发现意外丢失或改写 |
| 保存/加载前后的 ID | 检查保存文件是否复现配置好的编码 |
| 前导空格、制表符、CRLF/LF、空行、组合字符、emoji | 覆盖格式和 Unicode 边界情况 |
| 字面标记与真实对话/工具样本 | 检查内容与控制 token 的边界 |
| 极长片段和文档 | 暴露运行限制、内存增长及截断行为 |

比较 token 数时，明确分母：每个 UTF-8 字节、每个码点，还是每个文档。
跨语言比较最好使用平行或以其他方式匹配的内容。单一英语的“每词 token 数”
不是多语言指标。

如果分词器有意执行归一化，预期还原结果可能应是归一化文本，而非原始字符串。
某些解码器还会改变空白或省略特殊 token。先定义预期约定，再判断还原检查是否通过。

### 8.2 选择流程前需要回答的问题

1. 哪些变换负责选择文档，哪些会改变保留文档内部的文本？
2. 哪些规则必须在编码时共享，哪些只属于训练语料选择或增强？
3. 分词器训练配比是否充分覆盖中文、英语、代码、数学和真实部署领域？
4. 某条清洗规则是否可能抹掉复现输入所必需的信息？
5. 模型专属结论的依据是本版本文件、技术报告，还是较早的同系列模型？
6. 其他同学能否依靠保存的文件和清单复现分词器，而无需猜测隐藏的预处理？

课堂练习：运行第 7 节前，先预测 `Hello世界ABC`、`2026 1234567` 和
`Hi.\nNext` 的预分词片段，再解释为什么预分词片段一致，不代表 token ID
或 token 数一定一致。

## 9. 资料阅读指南

正文结论旁的链接均为一手资料。报告描述训练实践；固定版本文件支持编码结论；
流程和课堂设计建议单独标明。

| 资料 | 建议阅读内容 |
| --- | --- |
| [Kimi K3 技术报告][kimi-report] | 第 3.1、3.4 节：语料筛选、改写和长输入准备 |
| [GLM-5 技术报告][glm-report] | 第 2.2 节：网页分类器、代码元数据、数学/科学提取；属于较早同系列证据 |
| [DeepSeek-V4 技术报告][ds-report] | 第 4.1 节：数据构建与沿用的训练策略 |
| [DeepSeek-V3 技术报告][ds-v3-report] | 第 4.1 节：标点/换行 token 拆分与 FIM |
| [FineWeb 数据集卡][fineweb] | 公开描述的网页语料处理配方 |
| [DataTrove][datatrove] | 可执行的语料处理组件与示例 |
| [Dolma 论文][dolma-paper] | 公开的数据整理决策及其评估 |
| [OpenWebMath][openwebmath] | 数学内容为何需要专门的提取流程 |
| [Unicode 归一化规范][unicode] | 规范归一化与兼容归一化的区别 |
| [SentencePiece 归一化][sentencepiece-normalization] | 随分词器模型保存的归一化规则 |
| [Hugging Face 分词流程][hf-pipeline] | 归一化器、预分词器、学习模型和后处理器 |
| [Hugging Face 对话模板][hf-chat] | 消息序列化及避免重复特殊 token |

[kimi-report]: https://arxiv.org/html/2607.24653v1
[kimi-tokenizer]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tokenization_kimi.py
[kimi-ranks]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tiktoken.model
[kimi-config]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/tokenizer_config.json
[glm-repo]: https://github.com/zai-org/GLM-5
[glm-card]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/README.md
[glm-report]: https://arxiv.org/html/2602.15763v2
[glm-tokenizer]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/tokenizer.json
[glm-config]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/tokenizer_config.json
[glm-template]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/chat_template.jinja
[glm-model-config]: https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json
[ds-report]: https://arxiv.org/html/2606.19348v1
[ds-v3-report]: https://arxiv.org/html/2412.19437v2
[ds-tokenizer]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer.json
[ds-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/tokenizer_config.json
[ds-encoding]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/encoding/README.md
[fineweb]: https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md
[datatrove]: https://github.com/huggingface/datatrove
[dolma-format]: https://github.com/allenai/dolma/blob/main/docs/data-format.md
[dolma-dedup]: https://github.com/allenai/dolma/blob/main/docs/deduplication.md
[dolma-paper]: https://arxiv.org/abs/2402.00159
[openwebmath]: https://arxiv.org/abs/2310.06786
[unicode]: https://www.unicode.org/reports/tr15/
[sentencepiece-normalization]: https://github.com/google/sentencepiece/blob/master/doc/normalization.md
[sentencepiece-options]: https://github.com/google/sentencepiece/blob/master/doc/options.md
[hf-pipeline]: https://huggingface.co/docs/tokenizers/main/en/pipeline
[hf-trainers]: https://huggingface.co/docs/tokenizers/api/trainers
[hf-chat]: https://huggingface.co/docs/transformers/en/chat_templating
