import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================
# 1. 德州扑克 RL 环境
# ==============================================
class PokerRL_Env:
    def __init__(self):
        self.suits = ['♠', '♥', '♣', '♦']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.rank_map = {r: i for i, r in enumerate(self.ranks)}
        self.suit_map = {s: i for i, s in enumerate(self.suits)}
        self.deck = [(r, s) for r in self.ranks for s in self.suits]
        self.action_names = ["fold", "call", "raise_50", "raise_100", "raise_500"]

    def _get_card_val(self, c):
        return [self.rank_map[c[0]], self.suit_map[c[1]]]

    def get_state(self):
        res = []
        for c in self.my_hand: res.extend(self._get_card_val(c))
        for i in range(5):
            if i < len(self.community_cards): res.extend(self._get_card_val(self.community_cards[i]))
            else: res.extend([-1, -1])
        res.extend([self.pot / 1000.0, self.my_stack / 1000.0, self.opp_action / 1000.0])
        return np.array(res, dtype=np.float32)

    def reset(self):
        self.current_deck = self.deck.copy()
        random.shuffle(self.current_deck)
        self.my_hand = [self.current_deck.pop(), self.current_deck.pop()]
        self.opp_hand = [self.current_deck.pop(), self.current_deck.pop()]
        self.community_cards = []
        self.my_stack = 1000
        self.opp_stack = 1000
        self.pot = 20

        opp_res = self.evaluate_hand(self.opp_hand)
        level = opp_res[0]
        high_card = opp_res[1][0]

        if level >= 1 and high_card >= 11: # 一对 Q, K, A
            self.opp_action = 500
        elif level >= 1: # 其他对子
            self.opp_action = 100
        else: # 高牌
            self.opp_action = 10

        self.pot += self.opp_action
        return self.get_state()

    def step(self, action_idx, fixed_community=None):
        if fixed_community is not None:
            self.community_cards = fixed_community
        else:
            while len(self.community_cards) < 5:
                self.community_cards.append(self.current_deck.pop())

        my_power_res = self.evaluate_hand(self.my_hand + self.community_cards)
        opp_power_res = self.evaluate_hand(self.opp_hand + self.community_cards)

        if action_idx == 0: # Fold
            reward = 0 # 诱导模型尝试弃牌
        else:
            bet = [0, 10, 50, 100, 500][action_idx]
            if my_power_res > opp_power_res:
                reward = self.opp_action + 10
            elif my_power_res < opp_power_res:
                reward = -bet - self.opp_action
            else:
                reward = 0
        return reward / 100.0

    def evaluate_hand(self, cards):
        if not cards: return (0, [0])
        values = sorted([self.rank_map[c[0]] for c in cards], reverse=True)
        counts = Counter(values)
        count_values = sorted([(cnt, val) for val, cnt in counts.items()], reverse=True)
        if count_values[0][0] == 4: return (7, [count_values[0][1]])
        if count_values[0][0] == 3 and len(count_values) > 1 and count_values[1][0] >= 2:
            return (6, [count_values[0][1], count_values[1][1]])
        if count_values[0][0] == 3: return (3, [count_values[0][1]] + values[:2])
        if count_values[0][0] == 2 and len(count_values) > 1 and count_values[1][0] == 2:
            return (2, [count_values[0][1], count_values[1][1]] + values[:1])
        if count_values[0][0] == 2: return (1, [count_values[0][1]] + values[:3])
        return (0, values[:5])

# ==============================================
# 2. GRPO 策略网络
# ==============================================
class GRPO_Policy(nn.Module):
    def __init__(self, input_dim=17, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

# ==============================================
# 3. 训练逻辑
# ==============================================
def train_grpo():
    env = PokerRL_Env()
    policy = GRPO_Policy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    group_size = 128 # 👈 提高到 128，强制覆盖所有动作
    epochs = 10000
    entropy_coeff = 0.2 # 👈 提高熵权重

    print("开始‘破冰’训练：大组采样 (G=128) + 弃牌奖励优化...")

    for ep in range(epochs):
        state = env.reset()
        state_t = torch.FloatTensor(state)

        fixed_community = [env.current_deck.pop() for _ in range(5)]

        probs = policy(state_t)
        # 👈 增加保底概率，确保 fold 动作一定会被采样到
        dist = torch.distributions.Categorical(probs + 0.05)

        actions = dist.sample((group_size,))
        log_probs = dist.log_prob(actions)

        rewards = []
        for a in actions:
            r = env.step(a.item(), fixed_community=fixed_community)
            rewards.append(r)

        rewards = torch.FloatTensor(rewards)

        if rewards.std() < 1e-6:
            advantages = rewards - rewards.mean()
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        old_log_probs = log_probs.detach()
        new_probs = policy(state_t)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = new_dist.entropy().mean()
        loss = policy_loss - entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if ep % 200 == 0:
            test_s = [0, 0, 1, 1] + [4, 0, 5, 1, 6, 2, -1, -1, -1, -1] + [0.52, 1.0, 0.5]
            test_probs = policy(torch.FloatTensor(test_s))
            test_action = torch.argmax(test_probs).item()

            action_counts = torch.bincount(actions, minlength=5)
            print(f"Ep {ep:4d} | Loss: {loss.item():.4f} | Reward: {rewards.mean().item()*100:.1f} | Entropy: {entropy.item():.3f} | Opp: 500 -> My: {env.action_names[test_action]} | Actions: {action_counts.tolist()}")

if __name__ == "__main__":
    train_grpo()
