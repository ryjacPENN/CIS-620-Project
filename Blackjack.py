import random
from random import random, shuffle
import numpy as np
from typing import List


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
        value = 0
        num_aces = 0
        hand = []

        if player == -1:
            hand = self.dealer_cards
        else:
            hand = self.player_cards
        for card in hand:
            card_num = card[0]
            if card_num == "J" or card_num == "Q" or card_num == "K":
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

    def HandEvaluation2(self, hand):
        value = 0
        num_aces = 0

        for card in hand:
            card_num = card[0]
            if card_num == "J" or card_num == "Q" or card_num == "K":
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

        def __init__(self, insurance=False):
            """Kuhn node definitions."""
            self.infoSet: str = ""
            self.NUM_ACTIONS = BJTrainer().NUM_ACTIONS
            self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)

        def getStrategy(self, realizationWeight: float):
            """Get current information set mixed strategy through regret-matching."""
            normalizingSum: float = 0
            for a in range(self.NUM_ACTIONS):
                self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
                normalizingSum += self.strategy[a]
            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0:
                    self.strategy[a] /= normalizingSum
                else:
                    self.strategy[a] = 1 / self.NUM_ACTIONS
                self.strategySum[a] += realizationWeight * self.strategy[a]
            return self.strategy

        def getAverageStrategy(self):
            """Get average information set mixed strategy across all training iterations."""
            avgStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            normalizingSum: float = sum(self.strategySum)
            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0:
                    avgStrategy[a] = self.strategySum[a] / normalizingSum
                else:
                    avgStrategy[a] = 1 / self.NUM_ACTIONS
            return avgStrategy

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
        for i in range(iterations):
            """Shuffle cards. and give two cards to player and dealer"""
            game.initial()
            util += self.cfr(game, "", 1, game.player_cards.copy())

        # for n in self.nodeMap.values():
        #     print(n)

        with open("output_file1", "w") as file:
            for n in self.nodeMap.values():
                file.write(str(n) + "\n")

        print(f"Average game value: {util / iterations}")

    def cfr(
        self, game: BlackJackGame, history: str, p0: float, playercardlist: list
    ) -> float:
        """Counterfactual regret minimization iteration."""
        playersum = game.HandEvaluation2(playercardlist)

        # player =1 is the dealer, player =0 is the player, deck[0] is the unknown card, deck[1:3] is the known card
        # if history == "":
        infoSet: str = f"{game.dealer_cards[1][0]} {playersum} "
        # else:
        #     infoSet: str = f"{game.dealer_cards[1][0]} {game.player_cards[0][0]} {game.player_cards[1][0]} {playersum} {history}"

        """Return payoff for terminal states. """
        """player has black jack, if dealer has black jack, return 0, else return 1"""
        if (
            game.player_cards[0][0] in ["K", "Q", "J"]
            and game.player_cards[1][0] in ["A"]
        ) or (
            game.player_cards[0][0] in ["A"]
            and game.player_cards[1][0] in ["K", "Q", "J"]
        ):
            if not (
                (
                    game.dealer_cards[1][0] in ["A"]
                    and game.dealer_cards[0][0] in ["K", "Q", "J"]
                )
                or (
                    game.dealer_cards[1][0] in ["K", "Q", "J"]
                    and game.dealer_cards[0][0] in ["A"]
                )
            ):
                print("player bj")
                return 1
            else:
                print("tie")
                return 0

        if (
            game.dealer_cards[0][0] in ["K", "Q", "J"]
            and game.dealer_cards[1][0] in ["A"]
        ) or (
            game.dealer_cards[0][0] in ["A"]
            and game.dealer_cards[1][0] in ["K", "Q", "J"]
        ):
            print("dealer bj")
            return -1

        if playersum > 21:
            print("boom", playersum, history)
            return -1

        if len(history) > 0 and history[-1] == "P":
            """compare the sum of player and dealer"""
            dealersum = game.dealerdraw()
            print("dealer sum:", dealersum, "player sum:", playersum, history)
            if playersum > dealersum:
                return 1
            elif playersum == dealersum:
                return 0
            else:
                return -1

        """Get information set node or create it if nonexistant. """
        node = self.nodeMap.get(infoSet)
        if node is None:
            # # if we just start the game, and the known card of dealer is A/K/Q/J, we can buy insurance
            # if history == "" and game.dealer_cards[1][0] in ["A", "K", "Q", "J"]:
            #     node = self.Node(True)
            # else:
            node = self.Node()
            node.infoSet = infoSet
            self.nodeMap[infoSet] = node

        """For each action, recursively call cfr with additional history and probability. """
        strategy: float = node.getStrategy(p0)
        util: float = np.zeros(self.NUM_ACTIONS, dtype=float)
        nodeUtil: float = 0
        # if len(history) == 0:
        #     if game.dealer_cards[1][0] in ["A"]:
        #         for a in range(self.NUM_ACTIONS):
        #             nextHistory = history + ("I" if a == 0 else "U")
        #             util[a] = -self.cfr(game, nextHistory, p0 * strategy[a])
        #             nodeUtil += strategy[a] * util[a]
        #     else:
        #         history = "U"
        #         for a in range(self.NUM_ACTIONS):
        #             nextHistory = history + ("D" if a == 0 else "P")
        #             if a == 0:
        #                 game.player_cards.append(game.deck.pop())
        #             util[a] = -self.cfr(game, nextHistory, p0 * strategy[a])
        #             nodeUtil += strategy[a] * util[a]

        # else:
        for a in [0, 1]:
            nextHistory = history + ("D" if a == 0 else "P")
            # print(nextHistory)

            if a == 0:
                pcardlist = playercardlist.copy()
                pcardlist.append(game.deck.pop())
                util[a] = -self.cfr(game, nextHistory, p0 * strategy[a], pcardlist)
                nodeUtil += strategy[a] * util[a]
            else:
                pcardlist1 = playercardlist.copy()
                util[a] = -self.cfr(game, nextHistory, p0 * strategy[a], pcardlist1)
                nodeUtil += strategy[a] * util[a]
        """For each action, compute and accumulate counterfactual regret. """
        for a in range(self.NUM_ACTIONS):
            regret: float = util[a] - nodeUtil
            # print(p0, regret, strategy[a])
            node.regretSum[a] += regret

        return nodeUtil


def main():
    iterations: int = 10000
    trainer: BJTrainer = BJTrainer()
    trainer.train(iterations)


if __name__ == "__main__":
    main()
