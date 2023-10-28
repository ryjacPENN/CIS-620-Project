import random

def DeckCreator():
    card_list = []
    suits = ["Spades", "Clubs", "Hearts", "Diamonds"]
    face_cards = ["J", "Q", "K", "A"]

    for s in suits:
        for i in range(2, 11):
            card_list.append((s, i))
        for f in face_cards:
            card_list.append((s, f))

    random.shuffle(card_list)

    return card_list

class BlackjackDeck:
    def __init__(self, num_players):
        self.deck = DeckCreator()

        self.player_cards = [None] * num_players
        self.player_bets = [None] * num_players
        self.dealer_cards = [None] * num_players

        self.current_player = 0

    def Shuffle(self):
        self.deck = DeckCreator()

    def NextTurn(self):
        pass

    def Hit(self):
        pass