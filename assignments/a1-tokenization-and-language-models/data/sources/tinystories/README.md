---
license: cdla-sharing-1.0
task_categories:
- text-generation
language:
- en
---


Dataset containing synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary.

Described in the following paper: https://arxiv.org/abs/2305.07759. 

The models referred to in the paper were trained on TinyStories-train.txt  (the file tinystories-valid.txt can be used for validation loss). These models can be found on Huggingface, at roneneldan/TinyStories-1M/3M/8M/28M/33M/1Layer-21M.

Additional resources:
tinystories_all_data.tar.gz - contains a superset of the stories together with metadata and the prompt that was used to create each story.

TinyStoriesV2-GPT4-train.txt - Is a new version of the dataset that is based on generations by GPT-4 only (the original dataset also has generations by GPT-3.5 which are of lesser quality). It contains all the examples in TinyStories.txt which were GPT-4 generated as a subset (but is significantly larger).

Evaluation_prompts.yaml: List of prompts used to evaluate our models (see paper)