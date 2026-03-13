# Poker GRPO

这是一个基于 **GRPO (Group Relative Policy Optimization)** 算法的德州扑克强化学习项目。

## 项目结构

- `poker_grpo.py`: 核心训练脚本，包含环境定义、策略网络和 GRPO 训练逻辑。
- `poker_play_state.py`: 基于状态机的德州扑克博弈流程实现。
- `poker_play.py`: 基础博弈逻辑（规则驱动的固定策略）。
- `requirements.txt`: 项目依赖。
- `README.md`: 项目文档。

---

## 1. 纯净玩法：poker_play.py

`poker_play.py` 是一个**基于规则**的德州扑克模拟器。玩家和对手都使用固定的策略行动，不需要任何机器学习。

### 核心特性

1. **发牌系统**：52 张牌（4 种花色 × 13 种点数），每人 2 张私有牌，5 张公共牌。
2. **牌型评估**：自动评估从"高牌"到"同花顺"的 9 种牌型。
3. **固定策略**：
   - 牌型越好，下注越多（如：一对下 20，三条下 80，同花下 150）。
   - 面对对手加注时，如果己方牌力 >= 对手的 50%，则跟注，否则弃牌。

### 运行示例

```bash
python poker_play.py
```

### 游戏流程示例

```
==================================================
--- 新局开始 | 我的筹码: 1000 | 对手筹码: 1000 ---
💰 投入底池: 你出 10, 对手出 10 | 当前总底池: 20
我的手牌: 7♦ 3♣
对手手牌: ?? ??
公共牌: [等待亮牌]

[回车] 推进到下一阶段...

--- 阶段: Pre-flop ---
💰 投入底池: 你出 10, 对手出 10 | 当前总底池: 40

[回车] 推进到下一阶段...

--- 阶段: FLOP ---
我的手牌: J♥ 9♦
对手手牌: ?? ??
公共牌: 7♥ 10♠ 5♦
💰 投入底池: 你出 10, 对手出 10 | 当前总底池: 60

[回车] 推进到下一阶段...

--- 阶段: TURN ---
我的手牌: J♥ 9♦
公共牌: 7♥ 10♠ 5♦ J♦
👉 你加注到 20...
🤝 对手选择跟注 20
💰 投入底池: 你出 20, 对手出 20 | 当前总底池: 100

[回车] 推进到下一阶段...

--- 阶段: RIVER ---
我的手牌: J♥ 9♦
公共牌: 7♥ 10♠ 5♦ J♦ 6♣
👉 你加注到 20...
🤝 对手选择跟注 20
💰 投入底池: 你出 20, 对手出 20 | 当前总底池: 140

[回车] 推进到下一阶段...

--- SHOWDOWN ---
我的手牌: J♥ 9♦
对手手牌: A♠ 4♦
公共牌: 7♥ 10♠ 5♦ J♦ 6♣
我的牌型: 一对 | 对手牌型: 高牌
🎉 你赢了! 赢取底池: 140
```

---

## 2. 强化学习训练：poker_grpo.py

`poker_grpo.py` 是本项目的核心，使用 **GRPO** 算法训练一个神经网络策略，让 AI 自己学会如何打扑克。

### 核心设计

1. **状态表示**：将牌面信息编码为 17 维向量
   - 我的手牌：4 维（花色 + 点数）× 2
   - 公共牌：10 维（最多 5 张）
   - 底池/筹码/对手动作：3 维

2. **动作空间**：5 个离散动作
   - `0: fold`（弃牌）
   - `1: call`（跟注，10 筹码）
   - `2: raise_50`（加注到 50）
   - `3: raise_100`（加注到 100）
   - `4: raise_500`（加注到 500）

3. **奖励机制**：
   - 弃牌：固定惩罚
   - 跟注/加注：根据最终牌型对比结算输赢

### 训练示例

```bash
python poker_grpo.py
```

输出示例：

```
开始‘破冰’训练：大组采样 (G=128) + 弃牌奖励优化...
Ep    0 | Loss: -0.3256 | Reward: 15.9 | Entropy: 1.606 | Opp: 500 -> My: raise_100 | Actions: [26, 33, 26, 24, 19]
Ep  200 | Loss: -0.3241 | Reward: -167.3 | Entropy: 1.395 | Opp: 500 -> My: call | Actions: [21, 47, 27, 19, 14]
Ep 1000 | Loss: -0.3574 | Reward: -106.5 | Entropy: 1.417 | Opp: 500 -> My: call | Actions: [16, 46, 27, 22, 17]
...
Ep 9800 | Loss: -0.3459 | Reward: -81.2 | Entropy: 1.362 | Opp: 500 -> My: raise_100 | Actions: [12, 54, 30, 22, 10]
```

其中：
- **Loss**：策略网络的损失函数值
- **Reward**：该批次的平均奖励
- **Entropy**：策略的熵值（越高表示越不确定，越低表示越"自信"）
- **Opp: 500 -> My**: 测试时，对手下 500 块，AI 选择的动作
- **Actions**: 该批次 128 个样本中，各动作的分布

---

## 3. GRPO 算法详解

### 3.1 传统 PPO 的问题

在强化学习中，我们需要估计**优势函数（Advantage）**：

$$A(s, a) = Q(s, a) - V(s)$$

传统 PPO 使用一个额外的 **Critic 网络** $V(s)$ 来估计价值函数，但这引入了两个问题：
1. **需要同时训练两个网络**：Actor（策略网络）和 Critic（价值网络），计算量大。
2. **Critic 估计不准确**：尤其在扑克这种高方差环境里，Critic 很难学准。

### 3.2 GRPO 的核心思想

**GRPO（Group Relative Policy Optimization）** 由 DeepSeek 提出，核心思想是：

> **不需要 Critic！通过对同一状态下的多个动作进行组内对比，来计算相对优势。**

### 3.3 数学推导

**Step 1：组采样**

对于同一个状态 $s$，我们采样 $G$ 个不同的动作：
$$\{a_1, a_2, ..., a_G\} \sim \pi_\theta(\cdot|s)$$

**Step 2：计算组内相对优势**

假设这 $G$ 个动作分别获得的奖励为 $r_1, r_2, ..., r_G$，则优势函数定义为：

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}$$

其中：
- $\text{mean}(\mathbf{r}) = \frac{1}{G}\sum_{j=1}^{G} r_j$
- $\text{std}(\mathbf{r}) = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \text{mean}(\mathbf{r}))^2}$
- $\epsilon$ 是一个极小常数，防止除零

**Step 3：策略优化目标**

类似于 PPO，GRPO 也使用 Clipping：

$$L^{GRPO}(\theta) = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot A_i, \; \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right) \cdot A_i\right)\right]$$

其中 $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$ 将概率比值限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，防止一次更新步长过大。

### 3.4 一个具体的例子

假设在某一轮训练中：
- 状态 $s$ = "对手下注 500，我手里是一对 10"
- 我们采样了 4 个动作：$\{a_1=\text{fold}, a_2=\text{call}, a_3=\text{raise_100}, a_4=\text{raise_500}\}$
- 对应的奖励：$\{r_1=0, r_2=-510, r_3=-510, r_4=-510\}$（fold 不亏钱，其他动作输惨了）

**计算优势：**
- $\text{mean} = (-1530) / 4 = -382.5$
- $\text{std} \approx 220.5$

$$A_1 = \frac{0 - (-382.5)}{220.5} \approx 1.73 \quad (\text{fold 是最好的！})$$
$$A_2 = A_3 = A_4 = \frac{-510 - (-382.5)}{220.5} \approx -0.58 \quad (\text{都很差})$$

**策略更新：**
由于 $A_1 > 0$，模型会增加选择 $a_1$ (fold) 的概率；
由于 $A_2, A_3, A_4 < 0$，模型会减少选择这些动作的概率。

这样，AI 就学会了：**"面对 500 块的重注，手里是一对 10 时，弃牌是最优选择！"**

---

## 4. 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行基础玩法：
   ```bash
   python poker_play.py
   ```

3. 运行强化学习训练：
   ```bash
   python poker_grpo.py
   ```

---

## 5. 算法亮点

1. **无需 Critic**：GRPO 通过组内归一化直接计算优势，节省了一半的训练时间。
2. **高样本效率**：在扑克这种奖励方差极大的环境里，组内对比能有效过滤掉环境的随机噪声。
3. **稳定训练**：配合熵正则化（Entropy Regularization）和梯度裁剪（Gradient Clipping），可以避免策略坍缩。

---

## 6. 参考

- DeepSeek-R1: [GRPO: Group Relative Policy Optimization](https://github.com/deepseek-ai/DeepSeek-R1)
- OpenAI: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

