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
        self.player_hand_value = [0] * num_players

        self.dealer_cards = []
        
        self.player_bets = [0] * num_players
        self.player_money = [initial_money] * num_players

        self.current_player = 0
        self.current_turn = 0

    # Creates a new deck with every card to simulate a reset of the deck
    def Shuffle(self):
        self.deck = DeckCreator()

    # Goes to the next turn, shuffles the deck, resets bets and gives every player and the dealer two cards
    def NextTurn(self):
        self.current_turn += 1
        self.Shuffle()

        for i in range(len(self.player_cards)):
            self.player_cards[i] = []
            self.player_cards[i].append(self.deck.pop())
            self.player_cards[i].append(self.deck.pop())
            
            self.player_hand_value[i] = self.HandEvaluation(i)

            self.player_bets[i] = 0
            
        self.dealer_cards = []
        self.dealer_cards.append(self.deck.pop())
        self.dealer_cards.append(self.deck.pop())

    # Evaluates how many points a player's hand is worth (use -1 for dealer)
    def HandEvaluation(self, player):
        value = 0
        num_aces = 0
        hand = []

        if player == -1:
            hand = self.dealer_cards
        else:
            hand = self.player_cards[player]

        for card in hand:
            card_num = card[1]
            if type(card_num) == int:
                value += card_num
            elif card_num == "J" or card_num == "Q" or card_num == "K":
                value += 10
            else:
                num_aces += 1

        if num_aces > 0:
            value += (num_aces - 1)
            if (value + 11) <= 21:
                value += 11
            else:
                value += 1

        return value

    # Gives a player a new card
    def Hit(self):
        self.player_cards[self.current_player].append(self.deck.pop())
        self.player_hand_value[self.current_player] = self.HandEvaluation(self.current_player)

    # Passes to the next player
    def Stand(self):
        self.current_player += 1

    # Player bets an amount of money. Return false if a player doesn't have enough money for a bet
    def Bet(self, amount, player):
        curr_money = self.player_money[player] - amount

        if (curr_money >= 0):
            self.player_bets[player] = amount
            self.player_money[player] = curr_money
            return True
        else:
            return False

    # Returns a string of a specific card in a player's deck (use -1 for dealer)
    def DisplayCard(self, player_num, card_num):
        card_name = ""
        if player_num == -1:
            card_name = str(self.dealer_cards[card_num][1]) + " of " + self.dealer_cards[card_num][0]
        else:
            card_name = str(self.player_cards[player_num][card_num][1]) + " of " + self.player_cards[player_num][card_num][0]

        return card_name

    # Displays the current state of the game. Must input whether the dealer's cards should be revealed
    def DisplayCurrentGameState(self, revealDealerCards):
        print("##################################")
        print("Turn", self.current_turn)
        
        # Determine whether all dealer cards are revealed, and displays the known cards
        if revealDealerCards:
            dealer_string = "Dealer Cards: "
            for c in range(len(self.dealer_cards)):
                dealer_string += self.DisplayCard(-1, c)
                if c < len(self.dealer_cards) - 1:
                    dealer_string += " | "
            print(dealer_string)
        else:
            print("Dealer Cards :", self.DisplayCard(-1, 0), "| ?") # Hidden card is 0 index, revealed card is 1 index

        # Prints out each players card, money, and hand value
        for p in range(len(self.player_cards)):
            player_string = "Player " + str(p + 1) + " Cards: "
            for c in range(len(self.player_cards[p])):
                player_string += self.DisplayCard(p, c) + " | "

            player_string += "$" + str(self.player_money[p]) + " | "
            player_string += "Bet Amount: $" + str(self.player_bets[p]) + " | "
            player_string += "Hand Value: " + str(self.player_hand_value[p])
            print(player_string)

    # Function for playing the game of blackjack
    def PlayBlackJack(self):
        while True:
            # Go to next turn to initial configuration
            self.NextTurn()
            
            # Print out game configuration
            self.DisplayCurrentGameState(True)
            self.DisplayCurrentGameState(False)
            
            # Allow all players to bet
            for p in range(len(self.player_bets)):
                while True:
                    bet_input = input("Enter Player " + str(p + 1) + "'s bet: ")
                    if bet_input.isdigit():
                        if (self.Bet(int(bet_input), p)):
                            break
                    
                    print("INVALID BET")

            # Allow all players to hit or stand
            while True:
                self.DisplayCurrentGameState(False)

                curr_input = input("Player " + str(self.current_player + 1) + "'s turn: ").upper()

                if curr_input == "HIT":
                    self.Hit()
                elif curr_input == "STAND":
                    self.Stand()
                elif curr_input == "END":
                    print("END")
                    return True
                
                if self.current_player > len(self.player_cards):
                    break

            


game = BlackJackGame(2, 100)
game.PlayBlackJack()