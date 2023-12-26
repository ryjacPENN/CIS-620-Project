from Hearts import *


def calculate_scores(history):
    rounds = history.split(",")
    scores = [0, 0, 0, 0]

    for round_history in rounds:
        cards, winner = round_history.split("-")[:-1], int(round_history.split("-")[-1])
        for card in cards:
            rank, suit = card[:-1], card[-1]

            if suit == "H":
                scores[winner] -= 1
            elif rank == "13" and suit == "S":
                scores[winner] -= 13

    return scores


class HeartsTrainer:
    NUM_ACTIONS = 52
    nodeMap = dict()

    class Node:
        """Information set node class definition."""

        def __init__(self):
            """We need to use infoset to get the avaliable actions, so info set should looks like (player_cards(frozenset),"10S-5S-3S-2S-0(first round history and winner),9C-5C(second round, current player is the third player because the first player is the winner last round and he should follow the C suit or play any card if he does not have C suit)")"""
            self.infoSet = (frozenset(), "")
            self.NUM_ACTIONS = HeartsTrainer().NUM_ACTIONS
            self.regretSum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.strategySum = np.zeros(self.NUM_ACTIONS, dtype=float)
            self.currentplayer = 0
            self.currenthistory = ""

        def getplayerandhistory(self):
            """Get the player and history from the infoSet"""
            roundlist = self.infoSet[1].split(",")
            if len(roundlist) == 0:
                # The first round
                self.currentplayer = 0
                self.currenthistory = ""
            else:
                if len(roundlist[-1].split("-")) >= 4:
                    # The last round is finished, player is the winner of last round
                    self.currentplayer = roundlist[-1].split("-")[4]
                    self.currenthistory = ""
                else:
                    # The last round is not finished
                    if len(roundlist) == 1:
                        # The first round is not finished
                        CardNumPlayed = len(roundlist[-1].split("-"))
                        self.currentplayer = CardNumPlayed
                        self.currenthistory = roundlist[0]
                    else:
                        # we have at least two round, so we need to check the last round winner to dicide the current player
                        lastwinner = roundlist[-2].split("-")[4]
                        CardNumPlayed = len(roundlist[-1].split("-"))
                        self.currentplayer = (int(lastwinner) + CardNumPlayed) % 4
                        self.currenthistory = roundlist[-1]

        def getStrategy(self, realizationWeight: float):
            """Get current information set mixed strategy through regret-matching."""
            normalizingSum: float = 0
            availableActions = self.GetAvailableAction(
                self.currentplayer, self.currenthistory
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

        def getAverageStrategy(self):
            """Get average information set mixed strategy across all training iterations."""
            avgStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)
            availableActions = self.GetAvailableAction(
                self.currentplayer, self.currenthistory
            )  # need pass the player and the history
            normalizingSum: float = np.sum(
                [
                    self.strategySum[a]
                    for a in range(self.NUM_ACTIONS)
                    if availableActions[a] > 0
                ]
            )
            for a in range(self.NUM_ACTIONS):
                if normalizingSum > 0 and availableActions[a] > 0:
                    avgStrategy[a] = self.strategySum[a] / normalizingSum
                elif availableActions[a] > 0:
                    avgStrategy[a] = 1 / np.sum(availableActions)

            return avgStrategy

        # def getAverageStrategy(self):
        #     """Get average information set mixed strategy across all training iterations."""
        #     avgStrategy = np.zeros(self.NUM_ACTIONS, dtype=float)
        #     normalizingSum: float = sum(self.strategySum)
        #     for a in range(self.NUM_ACTIONS):
        #         if normalizingSum > 0:
        #             avgStrategy[a] = self.strategySum[a] / normalizingSum
        #         else:
        #             avgStrategy[a] = 1 / self.NUM_ACTIONS
        #     return avgStrategy

        def __str__(self):
            """Get information set string representation."""
            return f"{self.infoSet}: {self.getAverageStrategy()}"

    def __init__(self):
        pass

    # -------------------unmodified ↓----------------
    def train(self, iterations: int) -> None:
        game = HeartsGame()
        util: float = 0
        for i in range(iterations):
            """Shuffle cards. and give two cards to player and dealer"""
            game.NewHand()
            util += self.cfr(game, "", 1, 1, 1, 1)

        with open("output_file1", "w") as file:
            for n in self.nodeMap.values():
                file.write(str(n) + "\n")

        print(f"Average game value: {util / iterations}")

    def cfr(
        self, game: HeartsGame, history: str, p0: float, p1: float, p2: float, p3: float
    ) -> float:
        """Counterfactual regret minimization iteration."""
        num_rounds = len(history.split(","))

        if num_rounds == 13 and len(history.split(",")[-1].split("-")) == 5:
            # The game is finished
            return calculate_scores(history)

        roundlist = history.split(",")
        finishflag = False
        firstflag = False
        if len(roundlist) == 0:
            # The first round
            currentplayer = 0
            firstflag = True
        else:
            if len(roundlist[-1].split("-")) >= 4:
                # The last round is finished, player is the winner of last round
                currentplayer = roundlist[-1].split("-")[4]
                finishflag = True
            else:
                # The last round is not finished
                if len(roundlist) == 1:
                    # The first round is not finished
                    CardNumPlayed = len(roundlist[-1].split("-"))
                    currentplayer = CardNumPlayed
                else:
                    # we have at least two round, so we need to check the last round winner to dicide the current player
                    lastwinner = roundlist[-2].split("-")[4]
                    CardNumPlayed = len(roundlist[-1].split("-"))
                    currentplayer = (int(lastwinner) + CardNumPlayed) % 4

        infoSet: str = (game.player_cards[currentplayer], history)

        """Get information set node or create it if nonexistant. """
        node = self.nodeMap.get(infoSet)

        if node is None:
            node = self.Node()
            node.infoSet = infoSet
            self.nodeMap[infoSet] = node

        """For each action, recursively call cfr with additional history and probability. """
        strategy: float = node.getStrategy(p0)
        util: float = np.zeros(self.NUM_ACTIONS, dtype=float)
        nodeUtil: float = 0

        for a in range(self.NUM_ACTIONS):
            if strategy[a] > 0:
                if firstflag:
                    nextHistory = str(a % 13 + 2) + INVERSEDICT[a // 13]
                else:
                    if finishflag:
                        nextHistory = (
                            history + "," + str(a % 13 + 2) + INVERSEDICT[a // 13]
                        )
                    else:
                        thisroundhist = roundlist[-1].split("-")
                        if len(thisroundhist) >= 3:
                            # The current round will be finished after current player's turn
                            winner = 0
                            maxnum = 0
                            if len(roundlist) >= 2:
                                # if order maybe changed
                                previouswinner = roundlist[-2].split("-")[4]
                                for i in len(thisroundhist):
                                    if int(thisroundhist[i][:-1]) > maxnum:
                                        maxnum = int(thisroundhist[i][:-1])
                                        winner = (previouswinner + i) % 4
                                if a % 13 + 2 > maxnum:
                                    winner = currentplayer
                            else:
                                for i in len(thisroundhist):
                                    if int(thisroundhist[i][:-1]) > maxnum:
                                        maxnum = int(thisroundhist[i][:-1])
                                        winner = i
                                if a % 13 + 2 > maxnum:
                                    winner = currentplayer

                            nextHistory = (
                                history
                                + "-"
                                + str(a % 13 + 2)
                                + INVERSEDICT[a // 13]
                                + "-"
                                + str(winner)
                            )

                        else:
                            nextHistory = (
                                history + " - " + str(a % 13 + 2) + INVERSEDICT[a // 13]
                            )

                match currentplayer:
                    case 0:
                        util[a] = self.cfr(
                            game, nextHistory, p0 * strategy[a], p1, p2, p3
                        )[currentplayer]
                    case 1:
                        util[a] = self.cfr(
                            game, nextHistory, p0, p1 * strategy[a], p2, p3
                        )[currentplayer]
                    case 2:
                        util[a] = self.cfr(
                            game, nextHistory, p0, p1, p2 * strategy[a], p3
                        )[currentplayer]
                    case 3:
                        util[a] = self.cfr(
                            game, nextHistory, p0, p1, p2, p3 * strategy[a]
                        )[currentplayer]

                nodeUtil += strategy[a] * util[a]

        """For each action, compute and accumulate counterfactual regret. """
        for a in range(self.NUM_ACTIONS):
            regret: float = util[a] - nodeUtil
            node.regretSum[a] += regret

        return nodeUtil


def main():
    iterations: int = 10000
    trainer: HeartsTrainer = HeartsTrainer()
    trainer.train(iterations)


if __name__ == "__main__":
    main()
