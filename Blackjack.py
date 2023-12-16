from random import shuffle
import numpy as np
from tqdm import tqdm


def DeckCreator():
    card_list = []
    suits = ["S", "C", "H", "D"]
    face_cards = ["J", "Q", "K", "A"]

    for s in suits:
        for i in range(2, 11):
            card_list.append(str(i) + s)
        for f in face_cards:
            card_list.append((f + s))

    shuffle(card_list)

    return card_list


class BlackJackGame:
    def __init__(self, num_players, initial_money):
        self.deck = DeckCreator()
        self.player_cards = []
        self.player_hand_value = 0
        self.dealer_cards = []
        self.dealer_hand_value = 0

    def initial(self):
        self.deck = DeckCreator()
        self.dealer_cards = []
        self.dealer_cards.append(self.deck.pop())
        self.dealer_cards.append(self.deck.pop())
        self.player_cards = []
        self.player_cards.append(self.deck.pop())
        self.player_cards.append(self.deck.pop())
        self.dealer_hand_value = self.HandEvaluation(-1)
        self.player_hand_value = self.HandEvaluation(1)

    def dealerdraw(self):
        self.dealer_hand_value = self.HandEvaluation(-1)
        while self.dealer_hand_value < 17:
            self.dealer_cards.append(self.deck.pop())
            self.dealer_hand_value = self.HandEvaluation(-1)
        if self.dealer_hand_value > 21:
            return -1
        else:
            return self.dealer_hand_value

    # Creates a new deck with every card to simulate a reset of the deck
    def Shuffle(self):
        self.deck = DeckCreator()

    def HandEvaluation(self, player):
        if player == -1:
            hand = self.dealer_cards
        else:
            hand = self.player_cards
        return self.HandEvaluation2(hand)

    def HandEvaluation2(self, hand):
        value = 0
        num_aces = 0

        for card in hand:
            card_num = card[:-1]
            if card_num in {"J", "Q", "K"}:
                value += 10
            elif card_num != "A":
                value += int(card_num)
            else:
                num_aces += 1

        if num_aces > 0:
            value += num_aces - 1
            if (value + 11) <= 21:
                value += 11
            else:
                value += 1

        return value


class BJTrainer:
    """BJ and CFR problem definitions. P:PASS D:DRAW I:INSURANCE U:UNINSURANCE"""

    NUM_ACTIONS = 2
    # str -> Node
    nodeMap = dict()

    class Node:
        """Information set node class definition."""

        def __init__(self, insurance=False, info_set=""):
            """Kuhn node definitions."""
            self.infoSet: str = info_set
            self.NUM_ACTIONS = BJTrainer().NUM_ACTIONS
            self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)

        def getStrategy(self, realization_weight: float):
            """Get current information set mixed strategy through regret-matching."""
            normalizing_sum: float = 0
            for a in range(self.NUM_ACTIONS):
                self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
                normalizing_sum += self.strategy[a]
            for a in range(self.NUM_ACTIONS):
                if normalizing_sum > 0:
                    self.strategy[a] /= normalizing_sum
                else:
                    self.strategy[a] = 1 / self.NUM_ACTIONS
                self.strategySum[a] += realization_weight * self.strategy[a]
            return self.strategy

        def getAverageStrategy(self):
            """Get average information set mixed strategy across all training iterations."""
            avg_strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            normalizing_sum: float = sum(self.strategySum)
            for a in range(self.NUM_ACTIONS):
                if normalizing_sum > 0:
                    avg_strategy[a] = self.strategySum[a] / normalizing_sum
                else:
                    avg_strategy[a] = 1 / self.NUM_ACTIONS
            return avg_strategy

        def __str__(self):
            """Get information set string representation."""
            return f"{self.infoSet}: {self.getAverageStrategy()}"

    def __init__(self):
        pass

    def train(self, iterations: int) -> None:
        """Train Kuhn poker."""
        # cards: List[int] = [1, 2, 3]
        game = BlackJackGame(2, 1)
        util: float = 0
        for i in tqdm(range(iterations)):
            """Shuffle cards. and give two cards to player and dealer"""
            game.initial()
            util += self.cfr(game, f"({game.player_cards[0][:-1]})({game.player_cards[1][:-1]})", 1, game.player_cards.copy())

        with open("output_file", "w") as file:
            for n in self.nodeMap.values():
                file.write(str(n) + "\n")

        print(f"Average game value: {util / iterations}")

    def cfr(self, game: BlackJackGame, history: str, p0: float, playercardlist: list) -> float:
        """Counterfactual regret minimization iteration."""
        playersum = game.HandEvaluation2(playercardlist)

        # player =1 is the dealer, player =0 is the player, deck[0] is the unknown card, deck[1:3] is the known card
        dealer_open_value = game.dealer_cards[1][:-1]
        dealer_open_value = "10" if dealer_open_value in {"J", "Q", "K"} else dealer_open_value
        info_set: str = f"{dealer_open_value} {playersum}"

        """Return payoff for terminal states. """
        """player has black jack, if dealer has black jack, return 0, else return 1"""
        if (game.player_cards[0][:-1] in {"10", "K", "Q", "J"} and game.player_cards[1][:-1] == "A"
                or game.player_cards[0][:-1] == "A" and game.player_cards[1][:-1] in {"10", "K", "Q", "J"}):
            if (game.dealer_cards[1][:-1] == "A" and game.dealer_cards[0][:-1] in {"10", "K", "Q", "J"}
                    or game.dealer_cards[1][:-1] in {"10", "K", "Q", "J"} and game.dealer_cards[0][:-1] == "A"):
                print("push")
                return 0
            else:
                print("player bj")
                return 1

        if (game.dealer_cards[1][:-1] == "A" and game.dealer_cards[0][:-1] in {"10", "K", "Q", "J"}
            or game.dealer_cards[1][:-1] in {"10", "K", "Q", "J"} and game.dealer_cards[0][:-1] == "A"):
            """dealer has BJ"""
            print("dealer bj")
            return -1

        if playersum > 21:
            print("busted", playersum, history)
            return -1

        if len(history) > 0 and history[-1] == "P":
            """compare the sum of player and dealer"""
            dealersum = game.dealerdraw()
            if playersum > dealersum:
                print("player win", playersum, history, dealersum)
                return 1
            elif playersum == dealersum:
                print("push", playersum, history, dealersum)
                return 0
            else:
                print("player lose", playersum, history, dealersum)
                return -1

        """Get information set node or create it if nonexistant. """
        node = self.nodeMap.get(info_set)
        if node is None:
            # # if we just start the game, and the known card of dealer is A/K/Q/J, we can buy insurance
            # if history == "" and game.dealer_cards[1][0] in ["A", "K", "Q", "J"]:
            #     node = self.Node(True)
            # else:
            node = self.Node(info_set=info_set)
            self.nodeMap[info_set] = node

        """For each action, recursively call cfr with additional history and probability. """
        strategy: np.ndarray = node.getStrategy(p0)
        util: np.ndarray = np.zeros(self.NUM_ACTIONS, dtype=float)
        node_util: float = 0
        for a in range(self.NUM_ACTIONS):
            next_history = history + ("D" if a == 0 else "P")
            pcardlist = playercardlist.copy()
            if a == 0:
                pcardlist.append(game.deck.pop())
                next_history += f"({pcardlist[-1][:-1]})"
            util[a] = self.cfr(game, next_history, float(p0 * strategy[a]), pcardlist)
            node_util += strategy[a] * util[a]

        """For each action, compute and accumulate counterfactual regret. """
        for a in range(self.NUM_ACTIONS):
            regret: float = float(util[a] - node_util)
            node.regretSum[a] += regret

        return node_util


def main():
    iterations: int = 1000000
    trainer: BJTrainer = BJTrainer()
    trainer.train(iterations)


if __name__ == "__main__":
    main()
