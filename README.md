# Poker GRPO

这是一个基于 GRPO (Group Relative Policy Optimization) 算法的德州扑克强化学习项目。

## 项目结构

- `poker_grpo.py`: 核心训练脚本，包含环境定义、策略网络和 GRPO 训练逻辑。
- `poker_play_state.py`: 基于状态机的德州扑克博弈流程实现。
- `poker_play.py`: 基础博弈逻辑。
- `requirements.txt`: 项目依赖。

## 算法说明

本项目实现了 DeepSeek 提出的 GRPO 算法，主要特点包括：
- **组内相对优势**：通过对同一状态下的多个采样进行归一化来计算优势函数，无需 Critic 网络。
- **策略裁剪**：保留 PPO 的 Clipping 机制以确保训练稳定性。
- **熵正则化**：通过熵奖励鼓励模型探索，防止策略坍缩。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 开始训练：
   ```bash
   python poker_grpo.py

