import torch
import torch.nn.functional as F
import numpy as np


class PolicyDiscrete(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyDiscrete, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SACDiscrete:
    def __init__(self, gamma, tau, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy,
                 device):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device

        self.actor = PolicyDiscrete(state_dim, hidden_dim, action_dim).to(device)

        self.critic1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic1_target = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2_target = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        return action.item()

    def QTarget(self, rewards, next_states, dones):
        action_probs = self.actor(next_states)
        log_probs = torch.log(action_probs + 1e-8)
        entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
        q_value1 = self.critic1_target(next_states)
        q_value2 = self.critic2_target(next_states)
        q_value = torch.sum(action_probs * torch.min(q_value1, q_value2), dim=1, keepdim=True)
        q_target = rewards + self.gamma * (q_value + self.log_alpha.exp() * entropy) * (1 - dones)
        return q_target

    def soft_update(self, net, target_net):
        for net_para, target_net_para in zip(net.parameters(), target_net.parameters()):
            target_net_para.data.copy_((1.0 - self.tau) * target_net_para.data + self.tau * net_para.data)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        Q_target = self.QTarget(rewards, next_states, dones)
        critic1_loss = torch.mean(F.mse_loss(self.critic1(states).gather(1, actions), Q_target.detach()))
        critic2_loss = torch.mean(F.mse_loss(self.critic2(states).gather(1, actions), Q_target.detach()))
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        q_value = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-entropy * self.log_alpha.exp() - q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
