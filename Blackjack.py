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

class BlackJackGame:
    def __init__(self, num_players, initial_money):
        self.deck = DeckCreator()

        self.player_cards = [[]] * num_players
        self.dealer_cards = []
        
        self.player_bets = [0] * num_players
        self.player_money = [initial_money] * num_players

        self.current_player = 0
        self.current_turn = 0

    def Shuffle(self):
        self.deck = DeckCreator()

    def NextTurn(self):
        self.current_turn += 1
        self.Shuffle()

        for i in range(len(self.player_cards)):
            self.player_cards[i] = []
            self.player_cards[i].append(self.deck.pop())
            self.player_cards[i].append(self.deck.pop())
            
        self.dealer_cards = []
        self.dealer_cards.append(self.deck.pop())
        self.dealer_cards.append(self.deck.pop())

    def Hit(self):
        self.player_cards[self.current_player].append(self.deck.pop())

    def Stand(self):
        self.current_player += 1

    def Bet(self, amount):
        curr_money = self.player_money[self.current_player] - amount

        if (curr_money >= 0):
            self.player_bets[self.current_player] = amount
            self.player_money[self.current_player] = curr_money
            return True
        else:
            return False

    def DisplayCard(self, player_num, card_num):
        card_name = ""
        if player_num == -1:
            card_name = str(self.dealer_cards[card_num][1]) + " of " + self.dealer_cards[card_num][0]
        else:
            card_name = str(self.player_cards[player_num][card_num][1]) + " of " + self.player_cards[player_num][card_num][0]

        return card_name

    def DisplayCurrentGameState(self):
        print("##################################")
        print("Turn", self.current_turn)
        print("Dealer Cards :", self.DisplayCard(-1, 1), "and ?") # Hidden card is 0 index, revealed card is 1 index

        for p in range(len(self.player_cards)):
            print("Player", (p + 1), "Cards :", self.DisplayCard(p, 0), "and", self.DisplayCard(p, 1), ": $" + str(self.player_money[p]))

        print("Player " + str(self.current_player + 1) + "'s turn: ")

    def PlayBlackJack(self):
        # Start game to initial configuration
        self.NextTurn()

        # Print out game configuration
        self.DisplayCurrentGameState()



game = BlackJackGame(2, 100)
game.PlayBlackJack()