# RL reading guide for LLM post-training and agents

These optional resources support Lecture 10 (supervised fine-tuning),
Lecture 11 (preferences, rewards, and alignment), and Lecture 15 (agents and
tool use). See the [course schedule](schedule.md). They introduce no new
graded deliverable or deadline.

We organize the material around two training settings: **alignment and
reasoning through response generation**, and **interactive decision-making
by LLM agents**. Both can be part of LLM post-training and share RL algorithms.
Agent training adds environmental observations between decisions, tool use,
planning, recovery, and sometimes multi-agent coordination. Distillation and
preference optimization are complementary post-training methods.

## Core resources

| Resource | Type and authors | How to use it |
| --- | --- | --- |
| [Post-Training Course](https://rlhfbook.com/course) | Lectures and slides; Nathan Lambert | Follow the explanations alongside the book, especially RL implementation, reasoning, distillation, and tool use. |
| [Reinforcement Learning from Human Feedback](https://rlhfbook.com/) | Online textbook; Nathan Lambert | Main reference for rewards, policy gradients, reasoning, preference optimization, distillation, and tool-use training. Prefer the current web edition for recent additions. |
| [RLHF Book repository](https://github.com/natolambert/rlhf-book) | Book source, reference implementations, and teaching materials | Connect the equations to the implementations in `code/`, including PPO, REINFORCE, GRPO, RLOO, and DPO. Follow the repository's setup instructions. |
| [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches](https://www.marl-book.com/) | Textbook; Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer | Study coordination, games, and learning among multiple agents. Includes a free PDF, code, and slides. Its scope is general MARL, rather than specifically LLM orchestration. |
| [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/pdf/2509.02547) | Survey; Guibin Zhang and colleagues | Connect RL to planning, tool use, memory, reasoning, and agent environments. Use its taxonomy and references to select focused follow-up papers. [Abstract and versions](https://arxiv.org/abs/2509.02547). |

The first three links are complementary entry points to Lambert's project,
not three separate books. The final resource is a survey.

## Suggested reading path

1. **Build the post-training foundation.** Use Lambert Chapters 3–8 for the
   training pipeline, instruction tuning, rewards, RL, reasoning, and direct
   preference optimization. Pair Chapter 6 with the course's RL math and
   implementation lectures.
2. **Connect distillation to agents.** Read Chapter 12 for synthetic data and
   distillation, Chapter 13 for tool use, and Chapter 16 for evaluation.
   Chapter 12 includes on-policy distillation (OPD) and on-policy
   self-distillation (OPSD). Distinguish the source of supervision from the
   policy-update algorithm and the interaction setting.
3. **Study interactive agent training.** Read the agentic-RL survey's problem
   formulation and capability taxonomy. For one example, identify the policy,
   observations, actions, environment, reward, and evaluation budget.
4. **Add multi-agent depth when relevant.** Use the MARL book for the learning
   and coordination questions that arise with multiple agents. Several agents
   can share one model; multi-agent and multimodal are different concepts.

## Discussion prompts

Share questions and reading notes in the
[course discussion](https://github.com/baojian/llm-26-fall/discussions/320).

- Compare generating code and receiving a final test score with repeatedly
  editing code, running tests, and responding to failures. What changes in
  the trajectory, observations, and credit-assignment problem?
- What information could a teacher use during agent training that the student
  will not have at inference? How would you evaluate whether distillation
  improves task success as well as matching teacher behavior?
- When would multiple agents help? Distinguish training an orchestrator with
  fixed workers from jointly training several agents, and count total cost.

Related course material: the [MiMo RL archive guide](mimo-rl-archive.md) and
the [Agentic Large Language Models reading discussion](https://github.com/baojian/llm-26-fall/discussions/298).

Links and chapter mapping checked on November 1, 2026 (Asia/Shanghai).
