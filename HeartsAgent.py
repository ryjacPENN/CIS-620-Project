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
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetworkRNN(nn.Module):
    def __init__(self, n_features=108, n_actions=52):
        super(QNetworkRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_features, hidden_size=64, nonlinearity="relu", batch_first=True)
        self.fc1 = nn.Linear(in_features=64 + n_actions, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, observation, action):
        out, h = self.rnn(observation, None)
        x = torch.cat((out[:, -1, :], action), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HeartsAgent:
    def __init__(self):
        self.q_net = QNetwork()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1.5e-6)
        self.last_observation = [None, None, None, None]
        self.last_action = [None, None, None, None]
        self.losses = []
        self.epsilon = 0.5
        self.discount = 0.99
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
            p = torch.softmax(q_pred, 0).numpy()
            self.last_action[player] = np.random.choice(actions, p=p)
            return self.last_action[player]

    def learn(self, player: int, observation: np.ndarray, actions: List[int], reward: float, terminated: bool):
        if self.last_observation[player] is None:
            self.last_observation[player] = observation
            return
        q_input = np.zeros((1, 212))
        q_input[0, 0:160] = self.last_observation[player]
        q_input[0, self.last_action[player] + 160] = 1
        q_pred = self.q_net(torch.from_numpy(q_input).type(torch.float32)).reshape(1)
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
        agent.losses.clear()
        if agent.episodes == 50:
            agent.epsilon = 0.2
        elif agent.episodes == 200:
            agent.epsilon = 0.1
        elif agent.episodes == 500:
            agent.epsilon = 0.05
    plt.figure()
    plt.xlabel("Game Episodes")
    plt.ylabel("Prediction Loss")
    plt.plot(losses)
    plt.savefig('loss.png')
    print("Final average loss: ", losses[-1])


if __name__ == "__main__":
    main()
