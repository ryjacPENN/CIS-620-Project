from random import shuffle
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
        self.observations = None
        self.player_cards = [{"S": set(), "C": set(), "H": set(), "D": set()} for _ in range(4)]
        self.player_trick_value = [0, 0, 0, 0]
        self.player_scores = [0, 0, 0, 0]
        self.current_pile = []
        self.current_suit = None
        self.first_player = 0
        self.heart_broken = False
        self.NewHand()

    def NewHand(self):
        self.Shuffle()
        for i in range(4):
            for j in range(13):
                val, suit = self.deck.pop()
                if suit == "C" and val == 2:
                    self.first_player = i
                self.player_cards[i][suit].add(val)
                self.observations[i][DICT1[suit] * 13 + val + 106] = 1
        self.heart_broken = False

    def Shuffle(self):
        self.deck = DeckCreator()
        self.observations = [np.zeros(160, dtype=float) for _ in range(4)]
        # 52 for cards have been played, 52 for cards in current trick, 4 for current suit, 52 for cards in hand

    def PlayAction(self, player, card) -> bool:
        # play a card from the player's hand
        val, suit = card
        if val not in self.player_cards[player][suit]:
            print("Card not in player's hand")
            return False
        if self.current_suit is None:
            if suit == "H" and not self.heart_broken and (self.player_cards[player]["S"] or self.player_cards[player]["C"] or self.player_cards[player]["D"]):
                print("Cannot play heart because heart is not broken")
                return False
            self.current_suit = suit
            for i in range(4):
                self.observations[i][DICT1[suit] + 104] = 1
        elif self.player_cards[player][self.current_suit] and suit != self.current_suit:
            print("You must follow the suit")
            return False
        if suit == "H":
            self.heart_broken = True
        self.player_cards[player][suit].remove(val)
        self.observations[player][DICT1[suit] * 13 + val + 106] = 0
        self.current_pile.append((player, card))
        for i in range(4):
            self.observations[i][DICT1[suit] * 13 + val - 2] = 1
            self.observations[i][DICT1[suit] * 13 + val + 50] = 1
        return True

    # def PassCards(self, player, threecards):
    #     # pass cards to the left
    #     for card in threecards:
    #         self.player_cards[player][card[1]].remove(card[0])
    #         self.player_cards[(player + 1) % 4][card[1]].add(card[0])

    def EvaluatePile(self):
        winning_player = 0
        winning_value = 0
        pile_value = 0

        for player, (val, suit) in self.current_pile:
            if suit == self.current_suit and val > winning_value:
                winning_player = player
                winning_value = val
            if suit == "H":
                pile_value += 1
            elif suit == "S" and val == 12:
                pile_value += 13
            for i in range(4):
                self.observations[i][DICT1[suit] * 13 + val + 50] = 0
        for i in range(4):
            self.observations[i][DICT1[self.current_suit] + 104] = 0
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

    def GetAvailableAction(self, player, is_first=False) -> List[int]:
        strategy = []
        if is_first:
            strategy.append(DICT1["C"] * 13)  # The first must play 2C
            return strategy
        if self.current_suit is None:
            for suit in DICT1:
                if suit == "H" and not self.heart_broken and (self.player_cards[player]["S"] or self.player_cards[player]["C"] or self.player_cards[player]["D"]):
                    continue
                for val in self.player_cards[player][suit]:
                    strategy.append(DICT1[suit] * 13 + val - 2)
            return strategy
        if self.player_cards[player][self.current_suit]:
            for val in self.player_cards[player][self.current_suit]:
                strategy.append(DICT1[self.current_suit] * 13 + val - 2)
            return strategy
        for suit in DICT1:
            for val in self.player_cards[player][suit]:
                strategy.append(DICT1[suit] * 13 + val - 2)
        return strategy


def main():
    game = HeartsGame()
    game.PlayerGame()


if __name__ == "__main__":
    main()
