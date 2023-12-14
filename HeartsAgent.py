from Hearts import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, n_features=160, n_actions=52):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=n_features + n_actions, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class HeartsAgent:
    def __init__(self):
        self.q_net = QNetwork()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)
        self.last_observation = [None, None, None, None]
        self.last_action = [None, None, None, None]
        self.losses = []
        self.epsilon = 0.05
        self.discount = 0.9
        self.episodes = 0

    def act(self, player: int, observation: np.ndarray, actions: List[int]) -> int:
        self.last_observation[player] = observation
        if torch.rand(1) < self.epsilon:
            self.last_action[player] = np.random.choice(actions)
            return self.last_action[player]
        with torch.no_grad():
            q_input = np.zeros((len(actions), 212))
            for i, action in enumerate(actions):
                q_input[i, 0:160] = observation
                q_input[i, action + 160] = 1
            q_pred = self.q_net(torch.from_numpy(q_input).type(torch.float32)).reshape(len(actions))
            if self.episodes < 200:
                p = torch.softmax(q_pred, 0).numpy()
                self.last_action[player] = np.random.choice(actions, p=p)
            else:
                self.last_action[player] = actions[q_pred.numpy().argmax()]
            return self.last_action[player]

    def learn(self, player: int, observation: np.ndarray, actions: List[int], reward: float, terminated: bool):
        if self.last_observation[player] is None:
            self.last_observation[player] = observation
            return
        q_input = np.zeros(212)
        q_input[0:160] = self.last_observation[player]
        q_input[self.last_action[player] + 160] = 1
        q_pred = self.q_net(torch.from_numpy(q_input).type(torch.float32))
        if terminated:
            q_target = torch.Tensor([0])
        else:
            with torch.no_grad():
                q_target_input = np.zeros((len(actions), 212))
                for i, action in enumerate(actions):
                    q_target_input[i, 0:160] = observation
                    q_target_input[i, action + 160] = 1
                q_target = self.q_net(torch.from_numpy(q_target_input).type(torch.float32)).max().reshape(1)
        q_target = reward + self.discount * q_target
        loss = F.mse_loss(q_pred, q_target)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_observation[player] = observation
        if terminated:
            self.last_observation[player] = None
            self.last_action[player] = None


def main():
    game = HeartsGame()
    agent = HeartsAgent()
    losses = []
    for episode in tqdm(range(1, 1001)):
        game.NewHand()
        winning_player, pile_value = -1, 0
        for r in range(13):
            for i in range(4):
                player = (i + game.first_player) % 4
                actions = game.GetAvailableAction(player, r == 0 and i == 0)
                agent.learn(player, game.observations[player], actions, -pile_value if player == winning_player else 0, False)
                action = agent.act(player, game.observations[player], actions)
                card = (action % 13 + 2, INVERSEDICT[action // 13])
                if not game.PlayAction(player, card):
                    print(game.observations[player][108:160], actions, action, card)
                    raise Exception()
            winning_player, pile_value = game.EvaluatePile()
        for player in range(4):
            agent.learn(player, game.observations[player], [], -pile_value if player == winning_player else 0, True)
        agent.episodes = episode
        losses.append(np.mean(agent.losses))
        if agent.episodes == 20:
            agent.epsilon = 0.02
        elif agent.episodes == 100:
            agent.epsilon = 0.01
    plt.figure()
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
