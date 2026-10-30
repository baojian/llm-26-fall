---
language:
- zh
pipeline_tag: text-generation
license: apache-2.0
task_categories:
- text-generation
size_categories:
- 10B<n<100B
base_model:
- deepseek-ai/DeepSeek-R1
---



# **Chinese Fineweb Edu Dataset V2.1**          [[中文]](#chinese)    [[English]](#english)

<a id="english"></a>

<p align="center">
<img width="600px" alt="OpenCSG" src="./logo.png">
</p>

<p align="center"><a href="https://opencsg.com/models">[OpenCSG Community]</a>   <a href="https://github.com/yuyijiong/fineweb-edu-chinese">[👾github]</a>  <a href="https://cdn-uploads.huggingface.co/production/uploads/64c71b27d43e4dee51a8b31a/HU6vz21qKTEmUBCWqCFh9.jpeg">[wechat]</a>  <a href="https://twitter.com/OpenCsg">[Twitter]</a> </p>

</div>

[📖Technical Report](https://arxiv.org/abs/2501.08197)

The **Chinese Fineweb Edu Dataset V2.1** is an enhanced version of the V2 dataset, designed specifically for natural language processing (NLP) tasks in the education sector. This version introduces two new data sources, **map-cc** and **opencsg-cc**, and retains data with scores ranging from 2 to 3. The dataset entries are organized into different folders based on their scores, allowing for flexible selection of data according to time and computational power requirements during training.


# Expanded Data Sources
#### Key Features

1. **New Data Sources**:
   - **map-cc**
   - **opencsg-cc**
2. **Score-Based Data Organization**:
   - Data entries are categorized into different folders based on their scores:
     - **4-5**: High-quality educational content with clear and coherent writing.
     - **3-4**: Suitable educational content with some minor issues in coherence or relevance.
     - **2-3**: Potentially useful educational content with notable limitations.
3. **Data Volume**:
   - **4-5**: 70 GB, approximately 46 billion tokens, 17,790,513 lines.
   - **3-4**: 800 GB, approximately 530 billion tokens, 289,975,835 lines.
   - **2-3**: 1.4 TB, approximately 930 billion tokens, 649,842,063 lines.
4. **Flexible Training**:
   - The dataset organization allows for selective use of data based on the available time and computational resources.
   - Researchers and developers can choose specific score ranges to train their models, optimizing for different scenarios.

#### Data Distribution by Score

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <div>
    <p align="center">score: 4-5</p>
    <img width="300px" alt="experiment" src="./v21_45_source_stats.png">
  </div>
  <div>
    <p align="center">score: 3-4</p>
    <img width="300px" alt="experiment" src="./v21_34_source_stats.png">
  </div>
  <div>
    <p align="center">score: 2-3</p>
    <img width="300px" alt="experiment" src="./v21_23_source_stats.png">
  </div>
</div>



**We warmly invite developers and researchers interested in this field to follow and engage with the community, working together to advance the technology. Stay tuned for the open-source release of the dataset!**

## License Agreement

Usage of the Chinese Fineweb Edu dataset requires adherence to the OpenCSG Community License. The Chinese Fineweb Edu dataset supports commercial use. If you plan to use the OpenCSG model or its derivatives for commercial purposes, you must comply with the terms and conditions outlined in the OpenCSG Community License as well as the Apache 2.0 License. For commercial use, please send an email to lorraineg@opencsg.com and obtain permission.

<a id="chinese"></a>

<p>

</p>

[📖Technical Report](https://arxiv.org/abs/2501.08197)

# Chinese Fineweb Edu V2.1数据集介绍

<p align="center">
<img width="600px" alt="OpenCSG" src
="./logo.png">
</p>



<p align="center"><a href="https://opencsg.com/models">[OpenCSG 社区]</a>   <a href="https://github.com/yuyijiong/fineweb-edu-chinese">[👾github]</a>  <a href="https://cdn-uploads.huggingface.co/production/uploads/64c71b27d43e4dee51a8b31a/HU6vz21qKTEmUBCWqCFh9.jpeg">[微信]</a>  <a href="https://twitter.com/OpenCsg">[推特]</a> </p>



</div>
**Chinese Fineweb Edu Dataset V2.1** 是 V2 数据集的增强版本，专为教育领域的自然语言处理（NLP）任务设计和优化。此版本引入了两个新的数据源 **map-cc** 和 **opencsg-cc**，并保留了评分为 2 到 3 的数据。数据条目根据评分存储在不同的文件夹中，用户可以根据时间和计算资源的需求灵活选择训练数据。



## 数据筛选范围扩大

1. **新增数据源**：
   - **map-cc**
   - **opencsg-cc**
2. **基于评分的数据组织**：
   - 数据条目按评分存储在不同的文件夹中：
     - **4-5**：高质量的教育内容，写作清晰且连贯。
     - **3-4**：适合教育使用的内容，可能在连贯性或相关性方面存在一些小问题。
     - **2-3**：潜在有用的教育内容，但存在明显的局限性。
3. **数据量**：
   - **4-5**：70 GB，约 46 亿 tokens，17,790,513 行。
   - **3-4**：800 GB，约 530 亿 tokens，289,975,835 行。
   - **2-3**：1.4 TB，约 930 亿 tokens，649,842,063 行。
4. **灵活的训练**：
   - 数据集的组织允许用户根据可用时间和计算资源选择特定评分范围的数据进行训练，优化不同场景下的使用。

#### 按评分的数据分布

<div style="display: flex; justify-content: space-between; align-items: center; gap: 20px;">
  <div style="text-align: left;">
    <p>score: 4-5</p>
    <img width="300px" alt="experiment" src="./v21_45_source_stats.png">
  </div>
  <div style="text-align: center;">
    <p>score: 3-4</p>
    <img width="300px" alt="experiment" src="./v21_34_source_stats.png">
  </div>
  <div style="text-align: right;">
    <p>score: 2-3</p>
    <img width="300px" alt="experiment" src="./v21_23_source_stats.png">
  </div>
</div>

**我们诚邀对这一领域感兴趣的开发者和研究者关注和联系社区，共同推动技术的进步。敬请期待数据集的开源发布！**


## 许可协议
使用 Chinese Fineweb Edu V2数据集需要遵循 OpenCSG 社区许可证。Chinese Fineweb Edu V2数据集支持商业用途。如果您计划将 OpenCSG 模型或其衍生产品用于商业目的，您必须遵守 OpenCSG 社区许可证以及 Apache 2.0 许可证中的条款和条件。如用于商业用途，需发送邮件至 lorraineg@opencsg.com，并获得许可。

## Citation

```
@misc{yu2025opencsgchinesecorpusseries,
      title={OpenCSG Chinese Corpus: A Series of High-quality Chinese Datasets for LLM Training}, 
      author={Yijiong Yu and Ziyun Dai and Zekun Wang and Wei Wang and Ran Chen and Ji Pei},
      year={2025},
      eprint={2501.08197},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.08197}, 
}
```
