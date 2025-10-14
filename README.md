# Communication with Factorized Policy Gradients in Multi-Agent Deep Reinforcement Learning

This repository contains the official implementation of the paper  
**[Communication with Factorized Policy Gradients in Multi-Agent Deep Reinforcement Learning](https://link.springer.com/article/10.1007/s00521-025-11272-9)**,  
published in *Neural Computing and Applications*.

We provide implementations for **CFPG**, **FACMAC**, and **G2ANet**, extending the [FACMAC](https://github.com/oxwhirl/facmac) framework for continuous multi-agent reinforcement learning.

---

## 🔍 Overview

This codebase focuses on **continuous communication** with **factorized policy gradients**.  
It includes environments and training scripts to reproduce experiments from the paper.

---

## 🧩 Environments

### 1. Continuous Predator–Prey

A continuous variant of the [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs) *simple tag* task.

To create a **purely cooperative** setting:
- The prey is controlled by a heuristic policy that moves to the location farthest from the nearest predator.
- Cooperative agents (predators) receive a **team reward of +10** when any agent catches the prey.
- Otherwise, no reward is given.

For more details, please refer to the paper.

---

### 2. Multi-Agent MuJoCo (MAMuJoCo)

A benchmark extending OpenAI’s [MuJoCo](https://github.com/openai/mujoco-py) suite.

**MAMuJoCo** consists of a range of robotic control tasks where multiple agents, representing different components of a single robot, must cooperate to accomplish a shared objective.

For more information, please check the [Multi-Agent MuJoCo repository](https://github.com/schroederdewitt/multiagent_mujoco).

---

## ⚙️ How to Use

To reproduce the main experiments, simply run:

```bash
sh sbatch_journal.sh
```
