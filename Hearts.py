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
        for i in range(2, 15):
            card_list.append((i, s))
    shuffle(card_list)
    return card_list


class HeartsGame:
    def __init__(self):
        self.deck = None
        self.player_cards = [{"S": set(), "C": set(), "H": set(), "D": set()} for _ in range(4)]
        self.player_trick_value = [0] * 4
        self.player_scores = [0] * 4
        self.current_pile = []
        self.current_suit = None
        self.first_player = 0
        self.heart_broken = False
        self.NewHand()

    def NewHand(self):
        self.Shuffle()
        for i in range(4):
            for j in range(13):
                card = self.deck.pop()
                if card[1] == "C" and card[0] == 2:
                    self.first_player = i
                self.player_cards[i][card[1]].add(card[0])
        self.heart_broken = False

    def Shuffle(self):
        self.deck = DeckCreator()

    def PlayAction(self, player, card) -> bool:
        # play a card from the player's hand
        if card[0] not in self.player_cards[player][card[1]]:
            print("Card not in player's hand")
            return False
        if self.current_suit is None:
            if card[1] == "H" and not self.heart_broken:
                print("Cannot play heart because heart is not broken")
                return False
            self.current_suit = card[1]
        elif self.player_cards[player][self.current_suit] and card[1] != self.current_suit:
            print("You must follow the suit")
            return False
        if card[1] == "H":
            self.heart_broken = True
        self.player_cards[player][card[1]].remove(card[0])
        self.current_pile.append((player, card))
        return True

    def PassCards(self, player, threecards):
        # pass cards to the left
        for card in threecards:
            self.player_cards[player][card[1]].remove(card[0])
            self.player_cards[(player + 1) % 4][card[1]].add(card[0])

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
            elif card[1] == "S" and card[0] == 12:
                pile_value += 13

        self.player_trick_value[winning_player] += pile_value
        self.first_player = winning_player
        self.current_pile = []
        self.current_suit = None

        return winning_player, pile_value

    def UpdateScores(self):
        for i in range(len(self.player_scores)):
            if self.player_trick_value[i] == 26:
                self.player_scores[i] -= 26
            else:
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
        for _ in range(13):
            for i in range(4):
                self.PlayerInput((i + self.first_player) % 4)
            print(self.EvaluatePile())
            print(self.player_trick_value)
        self.UpdateScores()
        print(self.player_scores)

    # ------------------------------------------------------

    def GetAvailableAction(self, player, isFirst=False) -> np.ndarray:
        strategy = np.full(52, 0, dtype=float)
        if isFirst:
            strategy[DICT1["C"] * 13] = 1  # The first must play 2C
            return strategy
        if self.current_suit is None:
            for s in DICT1:
                if s == "H" and not self.heart_broken:
                    continue
                for card in self.player_cards[player][s]:
                    strategy[DICT1[s] * 13 + card - 2] = 1
            return strategy
        if self.player_cards[player][self.current_suit]:
            for card in self.player_cards[player][self.current_suit]:
                strategy[DICT1[self.current_suit] * 13 + card - 2] = 1
            return strategy
        for s in DICT1:
            for card in self.player_cards[player][s]:
                strategy[DICT1[s] * 13 + card - 2] = 1
        return strategy


def main():
    iterations: int = 10000
    # trainer: BJTrainer = BJTrainer()
    # trainer.train(iterations)
    game = HeartsGame()
    game.PlayerGame()


if __name__ == "__main__":
    main()
