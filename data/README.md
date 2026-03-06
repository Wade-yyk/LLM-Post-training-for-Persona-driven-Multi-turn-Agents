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
├── train/         
└── README.md

