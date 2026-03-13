import random
from collections import Counter
from enum import Enum, auto

class GameState(Enum):
    PRE_FLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()
    GAME_OVER = auto()

class RealPokerEnv:
    def __init__(self):
        self.suits = ['♠', '♥', '♣', '♦']
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.rank_values = {r: i for i, r in enumerate(self.ranks)}
        self.deck = [(r, s) for r in self.ranks for s in self.suits]

        # 筹码初始化
        self.my_stack = 1000
        self.opp_stack = 1000
        self.pot = 0
        self.current_state = GameState.GAME_OVER

        # 状态处理映射
        self.state_handlers = {
            GameState.PRE_FLOP: self._handle_pre_flop,
            GameState.FLOP: self._handle_flop,
            GameState.TURN: self._handle_turn,
            GameState.RIVER: self._handle_river,
            GameState.SHOWDOWN: self._handle_showdown
        }

    def reset(self):
        self.current_deck = self.deck.copy()
        random.shuffle(self.current_deck)
        self.my_hand = [self.current_deck.pop(), self.current_deck.pop()]
        self.opp_hand = [self.current_deck.pop(), self.current_deck.pop()]
        self.community_cards = []
        self.pot = 0
        self.current_state = GameState.PRE_FLOP

        print(f"\n{'='*50}")
        print(f"--- 新局开始 | 我的筹码: {self.my_stack} | 对手筹码: {self.opp_stack} ---")

        # 强制底注
        self.post_bet(10, 10)
        self.print_status()

    def post_bet(self, my_amt, opp_amt):
        """处理下注逻辑"""
        self.my_stack -= my_amt
        self.opp_stack -= opp_amt
        self.pot += (my_amt + opp_amt)
        print(f"💰 投入底池: 你出 {my_amt}, 对手出 {opp_amt} | 当前总底池: {self.pot}")

    def handle_betting_round(self):
        """处理下注轮，确保筹码配平或有人弃牌"""
        my_val = self.get_strategy_bet(self.my_hand + self.community_cards)
        opp_val = self.get_strategy_bet(self.opp_hand + self.community_cards)

        if my_val == opp_val:
            self.post_bet(my_val, opp_val)
            return True

        if my_val > opp_val:
            print(f"👉 你加注到 {my_val}...")
            if opp_val >= my_val * 0.5:
                print(f"🤝 对手选择跟注 {my_val}")
                self.post_bet(my_val, my_val)
                return True
            else:
                print(f"🏳️ 对手选择弃牌！")
                self.award_pot("Me")
                return False
        else:
            print(f"👉 对手加注到 {opp_val}...")
            if my_val >= opp_val * 0.5:
                print(f"🤝 你选择跟注 {opp_val}")
                self.post_bet(opp_val, opp_val)
                return True
            else:
                print(f"🏳️ 你选择弃牌！")
                self.award_pot("Opponent")
                return False

    def award_pot(self, winner):
        if winner == "Me":
            print(f"🎉 对手弃牌，你赢取底池: {self.pot}")
            self.my_stack += self.pot
        else:
            print(f"💀 你弃牌，对手赢取底池: {self.pot}")
            self.opp_stack += self.pot
        self.pot = 0
        self.current_state = GameState.GAME_OVER

    def step(self):
        """状态机驱动函数"""
        if self.current_state == GameState.GAME_OVER:
            print("游戏已结束，请重置。")
            return

        handler = self.state_handlers.get(self.current_state)
        if handler:
            handler()

    def _handle_pre_flop(self):
        print(f"\n--- 阶段: PRE-FLOP ---")
        if self.handle_betting_round():
            self.current_state = GameState.FLOP
        else:
            self.current_state = GameState.GAME_OVER

    def _handle_flop(self):
        print(f"\n--- 阶段: FLOP ---")
        for _ in range(3):
            self.community_cards.append(self.current_deck.pop())
        self.print_status()
        if self.handle_betting_round():
            self.current_state = GameState.TURN
        else:
            self.current_state = GameState.GAME_OVER

    def _handle_turn(self):
        print(f"\n--- 阶段: TURN ---")
        self.community_cards.append(self.current_deck.pop())
        self.print_status()
        if self.handle_betting_round():
            self.current_state = GameState.RIVER
        else:
            self.current_state = GameState.GAME_OVER

    def _handle_river(self):
        print(f"\n--- 阶段: RIVER ---")
        self.community_cards.append(self.current_deck.pop())
        self.print_status()
        if self.handle_betting_round():
            self.current_state = GameState.SHOWDOWN
        else:
            self.current_state = GameState.GAME_OVER

    def _handle_showdown(self):
        print(f"\n--- 阶段: SHOWDOWN ---")
        self.determine_winner()
        self.current_state = GameState.GAME_OVER

    def get_strategy_bet(self, cards):
        res = self.evaluate_hand(cards)
        bet_map = {
            0: 10,  # 高牌
            1: 20,  # 一对
            2: 50,  # 两对
            3: 80,  # 三条
            4: 100, # 顺子
            5: 150, # 同花
            6: 200, # 葫芦
            7: 300, # 四条
            8: 500  # 同花顺
        }
        return bet_map.get(res[0], 10)

    def evaluate_hand(self, cards):
        if not cards: return (0, 0)
        values = sorted([self.rank_values[c[0]] for c in cards], reverse=True)
        suits = [c[1] for c in cards]

        flush_suit = next((s for s in self.suits if suits.count(s) >= 5), None)
        unique_values = sorted(list(set(values)), reverse=True)
        straight_high = None
        if len(unique_values) >= 5:
            for i in range(len(unique_values) - 4):
                if unique_values[i] - unique_values[i+4] == 4:
                    straight_high = unique_values[i]
                    break
            if set([12, 0, 1, 2, 3]).issubset(set(unique_values)):
                straight_high = 3

        if flush_suit and straight_high: return (8, straight_high)

        counts = Counter(values)
        count_values = sorted([(cnt, val) for val, cnt in counts.items()], reverse=True)

        if count_values[0][0] == 4: return (7, count_values[0][1])
        if count_values[0][0] == 3 and len(count_values) > 1 and count_values[1][0] >= 2: return (6, count_values[0][1])
        if flush_suit: return (5, values[:5])
        if straight_high: return (4, straight_high)
        if count_values[0][0] == 3: return (3, count_values[0][1])
        if count_values[0][0] == 2 and len(count_values) > 1 and count_values[1][0] == 2: return (2, count_values[0][1])
        if count_values[0][0] == 2: return (1, count_values[0][1])
        return (0, values[:5])

    def determine_winner(self):
        my_res = self.evaluate_hand(self.my_hand + self.community_cards)
        opp_res = self.evaluate_hand(self.opp_hand + self.community_cards)

        hand_names = ["高牌", "一对", "两对", "三条", "顺子", "同花", "葫芦", "四条", "同花顺"]

        print(f"\n--- 最终摊牌 ---")
        self.print_status(show_opponent=True)
        print(f"我的牌型: {hand_names[my_res[0]]} | 对手牌型: {hand_names[opp_res[0]]}")

        if my_res > opp_res:
            print(f"🎉 你赢了! 赢取底池: {self.pot}")
            self.my_stack += self.pot
        elif my_res < opp_res:
            print(f"💀 对手赢了! 失去底池: {self.pot}")
            self.opp_stack += self.pot
        else:
            print(f"🤝 平局! 平分底池: {self.pot}")
            self.my_stack += self.pot // 2
            self.opp_stack += self.pot // 2
        self.pot = 0

    def print_status(self, show_opponent=False):
        def format_card(c): return f"{c[0]}{c[1]}"
        print(f"我的手牌: {' '.join([format_card(c) for c in self.my_hand])}")
        if show_opponent:
            print(f"对手手牌: {' '.join([format_card(c) for c in self.opp_hand])}")
        else:
            print(f"对手手牌: ?? ??")
        print(f"公共牌: {' '.join([format_card(c) for c in self.community_cards]) if self.community_cards else '[等待亮牌]'}")

if __name__ == "__main__":
    env = RealPokerEnv()
    for i in range(3):
        env.reset()
        while env.current_state != GameState.GAME_OVER:
            input(f"\n[回车] 推进到下一阶段...")
            env.step()
