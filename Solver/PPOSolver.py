import sys

if "../" not in sys.path:
  sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
from utils.util import compuate_GAE

class PolicyNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PolicyNet, self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc=torch.nn.Linear(input_dim,hidden_dim)
        self.fc_mu=torch.nn.Linear(hidden_dim,output_dim)
        self.fc_std=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc(x))
        mu=2.0*F.tanh(self.fc_mu(x))
        std=F.softplus(self.fc_std(x))
        return mu,std

class ValueNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class PPO:
    def __init__(self,gamma,lmbda,a_lr,c_lr,epsilon,epochs,input_dim,hidden_dim,output_dim,device):
        self.gamma=gamma
        self.lmbda=lmbda
        self.epsilon=epsilon
        self.epochs=epochs

        self.actor=PolicyNet(input_dim,hidden_dim,output_dim).to(device)
        self.critic=ValueNet(input_dim,hidden_dim).to(device)
        self.actorOptimizer=torch.optim.Adam(self.actor.parameters(),lr=a_lr)
        self.criticOptimizer=torch.optim.Adam(self.critic.parameters(),lr=c_lr)
        self.device=device

    def take_action(self,state):
        state=torch.tensor(np.array([state]),dtype=torch.float32).to(self.device)
        probs=self.actor(state)
        action_dist=torch.distributions.Categorical(probs=probs)
        action=action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float32).to(self.device)
        actions=torch.tensor(transition_dict['actions'],dtype=torch.float32).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float32).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float32).view(-1,1).to(self.device)

        TD_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        TD_delta=TD_target-self.critic(states)

        advantage=compuate_GAE(self.lmbda,self.gamma,TD_delta.cpu()).to(self.device)
        old_log_probs=torch.log(self.actor(states).gather(1,actions)).detach()

        for _ in range(self.epochs):
            log_probs=torch.log(self.actor(states).gather(1,actions))
            ratio=torch.exp(log_probs-old_log_probs)
            surr1=ratio*advantage
            surr2=torch.clamp(ratio,1-self.epsilon,1+self.epsilon)*advantage
            critic_loss=torch.mean(F.mse_loss(self.critic(states),TD_target.detach()))
            actor_loss=torch.mean(-torch.min(surr1,surr2))
            self.actorOptimizer.zero_grad()
            self.criticOptimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actorOptimizer.step()
            self.criticOptimizer.step()

class PPOContinuous:
    def __init__(self,gamma,lmbda,a_lr,c_lr,epsilon,epochs,input_dim,hidden_dim,output_dim,device):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.epochs = epochs

        self.actor = PolicyNetContinuous(input_dim, hidden_dim, output_dim).to(device)
        self.critic = ValueNet(input_dim, hidden_dim).to(device)
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.device = device

    def take_action(self,states):
        states=torch.tensor(np.array([states]),dtype=torch.float32).to(self.device)
        mu,std=self.actor(states)
        action_dist=torch.distributions.Normal(mu,std)
        action=action_dist.sample()
        return [action.item()]

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)

        rewards=(rewards+8.0)/8.0
        TD_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        TD_delta=TD_target-self.critic(states)

        advantage=compuate_GAE(self.lmbda,self.gamma,TD_delta.cpu()).to(self.device)
        mu,std=self.actor(states)
        action_dist=torch.distributions.Normal(mu.detach(),std.detach())
        old_log_probs=action_dist.log_prob(actions)

        for i in range(self.epochs):
            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(actions)
            ratio=torch.exp(log_probs-old_log_probs)
            surr1=ratio*advantage
            surr2=torch.clamp(ratio,1-self.epsilon,1+self.epsilon)*advantage

            actor_loss=torch.mean(-torch.min(surr1,surr2))
            critic_loss=torch.mean(F.mse_loss(self.critic(states),TD_target.detach()))
            self.actorOptimizer.zero_grad()
            self.criticOptimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actorOptimizer.step()
            self.criticOptimizer.step()
