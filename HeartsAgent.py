from Hearts import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_features=52, n_actions=52):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HeartsAgent:
    def __init__(self, player):
        self.game = HeartsGame()
        self.player = player
        self.q_net = QNetwork()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)
        self.last_observation = None
        self.last_action = None
        self.epsilon = 0.05
        self.discount = 0.9
        self.episodes = 0

    def act(self, observation: np.ndarray) -> int:
        """
        Compute the action given the observation
        Arguments:
            observation: an observation.
        Returns:
            An action.
        """
        actions = self.game.GetAvailableAction(self.player)
        self.last_observation = observation
        if torch.rand(1) < self.epsilon:
            self.last_action = np.random.choice(actions.size, p=actions)
            return self.last_action
        with torch.no_grad():
            q_pred = self.q_net(torch.from_numpy(observation).type(torch.float32))
            if self.episodes < 200:
                p = torch.softmax(q_pred, 0).numpy() * actions
                self.last_action = np.random.choice(actions.size, p=p)
            else:
                self.last_action = (q_pred.numpy() * actions).argmax().item()
            return self.last_action

    def learn(
        self,
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Do one step of Q-learning
        Arguments:
            observation: an observation
            reward: a reward
            terminated: whether the episode has terminated
            truncated: whether the episode was truncated
        """
        if self.last_observation is None:
            self.last_observation = observation
            return
        q_pred = self.q_net(
            torch.from_numpy(self.last_observation).type(torch.float32)
        )[self.last_action]
        with torch.no_grad():
            q_target = self.q_net(
                torch.from_numpy(observation).type(torch.float32)
            ).max()
            q_target = reward + self.discount * q_target
        loss = F.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_observation = observation
        if terminated or truncated:
            self.last_observation = None
            self.last_action = None
            self.episodes += 1
            if self.episodes == 20:
                self.epsilon = 0.02
            elif self.episodes == 100:
                self.epsilon = 0.01
