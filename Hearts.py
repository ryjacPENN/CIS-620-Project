import random
from random import random, shuffle
import numpy as np
from typing import List

DICT1 = {"S": 0, "C": 1, "H": 2, "D": 3}
INVERSEDICT = {0: "S", 1: "C", 2: "H", 3: "D"}


def DeckCreator():
    card_list = []
    suits = ["S", "C", "H", "D"]

    for s in suits:
        for i in range(2, 5):
            card_list.append((i, s))

    return card_list


class HeartsGame:
    def __init__(self):
        self.player_cards = [set() for i in range(4)]
        self.player_cards_hash = [frozenset() for i in range(4)]
        self.player_trick_value = [0] * 4
        self.player_scores = [0] * 4
        self.current_pile = []
        self.current_suit = None
        self.first_player = 0
        self.NewHand()

    def NewHand(self):
        self.player_cards = [set() for i in range(4)]
        self.player_cards_hash = [frozenset() for i in range(4)]
        self.player_trick_value = [0] * 4
        self.player_scores = [0] * 4
        self.current_pile = []
        self.current_suit = None
        self.first_player = 0
        self.Shuffle()
        for i in range(4):
            for j in range(5):
                self.player_cards[i].add(self.deck.pop())
        for i in range(4):
            self.player_cards_hash[i] = frozenset(self.player_cards[i])

    def Shuffle(self):
        self.deck = DeckCreator()

    def PlayAction(self, player, card):
        # play a card from the player's hand
        if card in self.player_cards[int(player)]:
            self.player_cards[int(player)].remove(card)
            self.current_pile.append((player, card))

            if self.current_suit == None:
                self.current_suit = card[1]

            return True

        else:
            print("Card not in player's hand")

            return False

    def ReversePlayAction(self, player, card):
        # play a card from the player's hand
        self.player_cards[int(player)].add(card)

    def PassCards(self, player, threecards):
        # pass cards to the left
        for card in threecards:
            self.player_cards[int(player)].remove(card)
            self.player_cards[int((player + 1) % 4)].append(card)

    def EvaluatePile(self):
        winning_player = 0
        winning_value = 0
        pile_value = 0

        for player, card in self.current_pile:
            if card[1] == self.current_suit and card[0] > winning_value:
                winning_player = player
                winning_value = card[0]
            if card[1] == "H":
                pile_value += 1
            elif card[1] == "S" and card[0] == 6:
                pile_value += 6

        self.player_trick_value[winning_player] += pile_value
        self.first_player = winning_player
        self.current_pile = []
        self.current_suit = None

        return winning_player, pile_value

    def UpdateScores(self):
        if 26 in self.player_trick_value:
            self.player_trick_value = [
                0 if self.player_trick_value[i] == 26 else 26 for i in range(4)
            ]
        for i in range(len(self.player_scores)):
            self.player_scores[i] += self.player_trick_value[i]
            self.player_trick_value[i] = 0

    # ------------------------------------------------------
    # ？ what are these 2 function doing ？
    def PlayerInput(self, player):
        while True:
            print("Current Player: " + str(player))
            print("Current Suit: " + str(self.current_suit))
            print("Current Pile: " + str(self.current_pile))
            print("Current Hand: " + str(self.player_cards[player]))
            card_str = input("Play Card: ")
            card = (int(card_str[0:-1]), card_str[-1])
            if self.PlayAction(player, card):
                break

    def PlayerGame(self):
        for i in range(3):
            for i in range(4):
                self.PlayerInput((i + self.first_player) % 4)
            print(self.EvaluatePile())
            print(self.player_trick_value)
        self.UpdateScores()
        print(self.player_scores)

    # ------------------------------------------------------

    def GetAvailableAction(self, player, thisroundhistory):
        strategy = np.full(20, 0, dtype=float)
        # eg. 10S-5S-3S-2S-1(first round history and winner)
        if len(thisroundhistory) == 0:
            # The current player plays first, He can play any card (ignoring the heart limit for now)
            for i in self.player_cards[int(player)]:
                strategy[i[0] - 2 + 5 * DICT1[i[1]]] = 1
            return strategy
        # eg. [10s,5s,3s,2s,1]
        else:
            thisroundhistory = thisroundhistory.split("-")
            # The current player is not the first player
            # The current player must follow the suit
            # If the current player does not have the suit, he can play any card
            # If the current player has the suit, he can only play the cards with the suit
            suit = thisroundhistory[0][-1]
            # eg. suit = "S"
            for i in self.player_cards[int(player)]:
                if i[1] == suit:
                    strategy[i[0] - 2 + 5 * DICT1[i[1]]] = 1
            if np.sum(strategy) == 0:
                for i in self.player_cards[int(player)]:
                    strategy[i[0] - 2 + 5 * DICT1[i[1]]] = 1
            return strategy


def calculate_scores(history):
    rounds = history.split(",")
    scores = [0, 0, 0, 0]
    if rounds[0] == "":
        rounds = []

    for round_history in rounds:
        if round_history.split("-")[0] != "":
            cards, winner = round_history.split("-")[:-1], int(
                round_history.split("-")[-1]
            )
            for card in cards:
                rank, suit = card[:-1], card[-1]

                if suit == "H":
                    scores[winner] -= 1
                elif rank == "6" and suit == "S":
                    scores[winner] -= 6

    return scores


class BJTrainer:
    NUM_ACTIONS = 20
    nodeMap = dict()

    class Node:
        """Information set node class definition."""

        def __init__(self):
            """We need to use infoset to get the avaliable actions, so info set should looks like (player_cards(frozenset),"10S-5S-3S-2S-0(first round history and winner),9C-5C(second round, current player is the third player because the first player is the winner last round and he should follow the C suit or play any card if he does not have C suit)")"""
            self.infoSet = (frozenset(), "")
            self.NUM_ACTIONS = BJTrainer().NUM_ACTIONS
            self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)

        # def getplayerandhistory(self):
        #     """Get the player and history from the infoSet"""
        #     roundlist = self.infoSet[1].split(",")
        #     if roundlist[0]=="":
        #       roundlist=[]

        #     if len(roundlist) == 0:
        #         # The first round
        #         self.currentplayer = 0
        #         self.currenthistory = ""
        #     else:
        #         if len(roundlist[-1].split("-")) >= 4:
        #             # The last round is finished, player is the winner of last round
        #             self.currentplayer = roundlist[-1].split("-")[4]
        #             self.currenthistory = ""
        #         else:
        #             # The last round is not finished
        #             if len(roundlist) == 1:
        #                 # The first round is not finished
        #                 CardNumPlayed = len(roundlist[-1].split("-"))
        #                 self.currentplayer = CardNumPlayed
        #                 self.currenthistory = roundlist[0]
        #             else:
        #                 # we have at least two round, so we need to check the last round winner to dicide the current player
        #                 lastwinner = roundlist[-2].split("-")[4]
        #                 CardNumPlayed = len(roundlist[-1].split("-"))
        #                 self.currentplayer = (int(lastwinner) + CardNumPlayed) % 4
        #                 self.currenthistory = roundlist[-1]

        def getStrategy(
            self, HeartsGame, realizationWeight: float, currentplayer, currenthistory
        ):
            """Get current information set mixed strategy through regret-matching."""
            normalizingSum: float = 0
            availableActions = HeartsGame.GetAvailableAction(
                currentplayer, currenthistory
            )  # need pass the player and the history

            for a in range(self.NUM_ACTIONS):
                if availableActions[a] > 0:
                    self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
                    normalizingSum += self.strategy[a]
                else:
                    self.strategy[a] = 0

            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0 and availableActions[a] > 0:
                    self.strategy[a] /= normalizingSum
                elif availableActions[a] > 0:
                    self.strategy[a] = 1 / np.sum(availableActions)
                self.strategySum[a] += realizationWeight * self.strategy[a]

            return self.strategy

        # def getStrategy(self, realizationWeight: float):
        #     """Get current information set mixed strategy through regret-matching."""
        #     normalizingSum: float = 0
        #     for a in range(self.NUM_ACTIONS):
        #         self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
        #         normalizingSum += self.strategy[a]
        #     for a in range(self.NUM_ACTIONS):
        #         if normalizingSum > 0:
        #             self.strategy[a] /= normalizingSum
        #         else:
        #             self.strategy[a] = 1 / self.NUM_ACTIONS
        #         self.strategySum[a] += realizationWeight * self.strategy[a]
        #     return self.strategy

        # def getAverageStrategy(self,HeartsGame):
        #     """Get average information set mixed strategy across all training iterations."""
        #     avgStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)
        #     availableActions = HeartsGame.GetAvailableAction(
        #         self.currentplayer, self.currenthistory
        #     )  # need pass the player and the history
        #     normalizingSum: float = np.sum(
        #         [
        #             self.strategySum[a]
        #             for a in range(self.NUM_ACTIONS)
        #             if availableActions[a] > 0
        #         ]
        #     )
        #     for a in range(self.NUM_ACTIONS):
        #         if normalizingSum > 0 and availableActions[a] > 0:
        #             avgStrategy[a] = self.strategySum[a] / normalizingSum
        #         elif availableActions[a] > 0:
        #             avgStrategy[a] = 1 / np.sum(availableActions)

        #     return avgStrategy

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
        util: float = 0
        for i in range(iterations):
            game = HeartsGame()
            game.NewHand()
            print("num", i, game.player_cards)
            print(str(i) + "th iteration")
            """Shuffle cards. and give two cards to player and dealer"""
            cfrs = self.cfr(game, "", 1, 1, 1, 1)
            print("util+" + str(cfrs))
            util += cfrs
            print(f"Average game value of {i+1}th iteration: {util / (i + 1)}")

        with open("output_file2", "w") as file:
            for n in self.nodeMap.values():
                file.write(str(n) + "\n")

        print(f"Average game value: {util / iterations}")

    def cfr(
        self, game: HeartsGame, history: str, p0: float, p1: float, p2: float, p3: float
    ) -> float:
        """Counterfactual regret minimization iteration."""
        num_rounds = len(history.split(","))

        if num_rounds == 5 and len(history.split(",")[-1].split("-")) == 5:
            # The game is finished
            # print("finish", history)
            return calculate_scores(history)[
                (int(history.split(",")[-2].split("-")[-1]) + 3) % 4
            ]

        roundlist = history.split(",")
        if roundlist[0] == "":
            roundlist = []
        finishflag = False
        firstflag = False
        if len(roundlist) == 0:
            # The first round
            currentplayer = 0
            firstflag = True
            currenthistory = ""
        else:
            if len(roundlist[-1].split("-")) >= 4:
                # The last round is finished, player is the winner of last round
                currentplayer = roundlist[-1].split("-")[4]
                finishflag = True
                currenthistory = ""
            else:
                # The last round is not finished
                if len(roundlist) == 1:
                    # The first round is not finished
                    CardNumPlayed = len(roundlist[-1].split("-"))
                    currentplayer = CardNumPlayed
                    currenthistory = roundlist[0]
                else:
                    # we have at least two round, so we need to check the last round winner to dicide the current player
                    lastwinner = roundlist[-2].split("-")[4]
                    CardNumPlayed = len(roundlist[-1].split("-"))
                    currentplayer = (int(lastwinner) + CardNumPlayed) % 4
                    currenthistory = roundlist[-1]

        # print("player:", currentplayer, "history", currenthistory)
        infoSet: str = (game.player_cards_hash[int(currentplayer)], history)

        """Get information set node or create it if nonexistant. """
        node = self.nodeMap.get(infoSet)

        if node is None:
            node = self.Node()
            node.infoSet = infoSet
            self.nodeMap[infoSet] = node

        """For each action, recursively call cfr with additional history and probability. """
        if currentplayer == 0:
            prob_players = p0
        elif currentplayer == 1:
            prob_players = p1
        elif currentplayer == 2:
            prob_players = p2
        else:  # currentplayer == 3
            prob_players = p3
        strategy: float = node.getStrategy(
            game, prob_players, currentplayer, currenthistory
        )
        util: float = np.zeros(self.NUM_ACTIONS, dtype=float)
        nodeUtil: float = 0
        print("stratgy!!!!", strategy)
        for a in range(self.NUM_ACTIONS):
            if strategy[a] > 0:
                if firstflag:
                    nextHistory = str(a % 5 + 2) + INVERSEDICT[a // 5]
                else:
                    if finishflag:
                        nextHistory = (
                            history + "," + str(a % 5 + 2) + INVERSEDICT[a // 5]
                        )
                    else:
                        thisroundhist = roundlist[-1].split("-")
                        if len(thisroundhist) >= 3:
                            # The current round will be finished after current player's turn
                            winner = 0
                            maxnum = 0
                            if len(roundlist) >= 2:
                                # if order maybe changed
                                previouswinner = int(roundlist[-2].split("-")[4])
                                for i in range(len(thisroundhist)):
                                    if int(thisroundhist[i][:-1]) > maxnum:
                                        maxnum = int(thisroundhist[i][:-1])
                                        winner = (previouswinner + i) % 4
                                if a % 5 + 2 > maxnum:
                                    winner = currentplayer
                            else:
                                for i in range(len(thisroundhist)):
                                    # print(thisroundhist)
                                    if int(thisroundhist[i][:-1]) > maxnum:
                                        maxnum = int(thisroundhist[i][:-1])
                                        winner = i
                                if a % 5 + 2 > maxnum:
                                    winner = currentplayer

                            nextHistory = (
                                history
                                + "-"
                                + str(a % 5 + 2)
                                + INVERSEDICT[a // 5]
                                + "-"
                                + str(winner)
                            )

                        else:
                            nextHistory = (
                                history + "-" + str(a % 5 + 2) + INVERSEDICT[a // 5]
                            )
                game.PlayAction(currentplayer, (a % 5 + 2, INVERSEDICT[a // 5]))
                match int(currentplayer):
                    case 0:
                        util[a] = -self.cfr(
                            game, nextHistory, p0 * strategy[a], p1, p2, p3
                        )
                    case 1:
                        util[a] = -self.cfr(
                            game, nextHistory, p0, p1 * strategy[a], p2, p3
                        )
                    case 2:
                        util[a] = -self.cfr(
                            game, nextHistory, p0, p1, p2 * strategy[a], p3
                        )
                    case 3:
                        util[a] = -self.cfr(
                            game, nextHistory, p0, p1, p2, p3 * strategy[a]
                        )
                # dfs need reverse
                game.ReversePlayAction(currentplayer, (a % 5 + 2, INVERSEDICT[a // 5]))
                nodeUtil += strategy[a] * util[a]

        """For each action, compute and accumulate counterfactual regret. """

        if currentplayer == 0:
            prob_other_players = p1 * p2 * p3
        elif currentplayer == 1:
            prob_other_players = p0 * p2 * p3
        elif currentplayer == 2:
            prob_other_players = p0 * p1 * p3
        else:  # currentplayer == 3
            prob_other_players = p0 * p1 * p2

        for a in range(self.NUM_ACTIONS):
            if strategy[a] > 0:
                regret: float = util[a] - nodeUtil
                node.regretSum[a] += (prob_other_players) * regret

        return nodeUtil


def main():
    iterations: int = 100000
    trainer: BJTrainer = BJTrainer()
    trainer.train(iterations)
    # game = HeartsGame()
    # game.PlayerGame()


main()
