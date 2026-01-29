LLM Post-training for Persona-driven Multi-turn Agents

Overview

This project explores LLM post-training strategies for persona-driven multi-turn agents, motivated by the goal of bringing fictional characters into interactive, real-world conversational systems.

Starting from a strong interest in ACG-style characters, this work focuses on instruction tuning and data-centric post-training to enable large language models to maintain consistent persona, tone, and behavior across long multi-turn conversations.

The project emphasizes practical post-training pipelines, controllable behavior design, and qualitative evaluation of agent consistency.


Motivation

Many existing LLM-based chat systems struggle with:

Persona drift in long conversations

Inconsistent tone or character behavior

Weak controllability under multi-turn interaction

Inspired by character-driven narratives in ACG works, this project treats persona as a controllable conditioning signal and investigates how instruction design and post-training strategies affect agent behavior stability.

Rather than focusing on pretraining from scratch, the project targets post-training techniques that are feasible, reproducible, and relevant to real-world applications such as AI NPCs and interactive agents.


Methodology Instruction Design

Manually constructed ~500 high-quality instruction–output pairs

Each instruction explicitly conditions:

Character persona

Speaking style and tone

Multi-turn conversational context

Instructions are designed to encourage:

Consistent persona adherence

Natural multi-turn dialogue flow

Context-aware responses

Raw copyrighted texts are not redistributed. Only instruction-style transformed data and schemas are used.

Model and Training


Base models:

Qwen-7B

Qwen-14B

Post-training method:

LoRA-based supervised fine-tuning

Multi-turn conversation training


Training focus:

Stability during long-context fine-tuning

Comparison between single-turn and multi-turn instruction setups

Qualitative analysis of persona consistency

The training pipeline is designed to be extensible to preference-based optimization (e.g., RLHF-style methods) in future work.


Project Structure
.
├── data/         
├── training/    
├── agent/       
├── evaluation/  
└── README.md


Evaluation

Evaluation focuses primarily on qualitative and behavioral metrics, including:

Persona consistency across long conversations

Tone and style adherence

Multi-turn coherence and memory retention

Future work will incorporate automated preference models and reinforcement learning–based evaluation.


Key Takeaways

Data quality and instruction structure have a significant impact on persona stability

Multi-turn instruction tuning introduces unique challenges compared to single-turn SFT

LoRA-based post-training provides an efficient and practical approach for controllable agent behavior

Reproducibility and Data Policy

This repository does not release any copyrighted or proprietary raw texts.


We provide:

Instruction schemas

Data generation scripts

Training and evaluation code

Users should prepare their own raw data from sources they are authorized to use.


Future Work:

Preference optimization using PPO / GRPO-style methods

Reward modeling for persona adherence

Integration with agent-based frameworks (memory, goals, environment interaction)

Evaluation in interactive game or NPC-like environments


License

MIT License
