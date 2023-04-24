import sys
if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay
import torch
import torch.nn.functional as F
import numpy as np


class QNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VAnet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.V = torch.nn.Linear(hidden_dim, 1)
        self.A = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        V = self.V(x)
        A = self.A(x)
        return V + A - A.mean(1).view(-1, 1)


class DQNs:
    def __init__(self, epsilon, gamma, learning_rate, input_dim, hidden_dim, output_dim, target_update, device,
                 dqn_type):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.device = device
        self.target_update = target_update
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.count = 0
        self.dqn_type = dqn_type
        if dqn_type == 'dueling_dqn':
            self.qnet = VAnet(input_dim, hidden_dim, output_dim).to(device)
            self.qnet_target = VAnet(input_dim, hidden_dim, output_dim).to(device)
        else:
            self.qnet = QNet(input_dim, hidden_dim, output_dim).to(device)
            self.qnet_target = QNet(input_dim, hidden_dim, output_dim).to(device)

        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.output_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
            if self.dqn_type == 'dueling_dqn' or self.dqn_type == 'double_dqn':
                action = self.qnet(state).argmax().item()
            else:
                action = self.qnet_target(state).argmax().item()
        return action

    def max_q(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        return self.qnet_target(state).max(1)[0].item()

    def update(self, transition):
        states = torch.tensor(transition['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition['actions']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition['next_states'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transition['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        dones = torch.tensor(transition['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        q_values = self.qnet(states).gather(1, actions)
        if self.dqn_type == 'dueling_dqn' or self.dqn_type == 'double_dqn':
            a = self.qnet(next_states).argmax(1).view(-1, 1)
            q_max_values = self.qnet_target(next_states).gather(1, a)
        else:
            q_max_values = self.qnet_target(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * q_max_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.count += 1