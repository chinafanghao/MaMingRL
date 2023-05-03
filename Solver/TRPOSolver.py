import torch
import torch.nn.functional as F
import numpy as np
from utils.util import compuate_GAE

class CriticNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(CriticNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class Policy(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Policy, self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1)
        return F.softmax(self.fc2(x),dim=1)

class TRPO:
    def __init__(self,gamma,lmbda,input_dim,hidden_dim,action_space,lr,device):
        self.gamma=gamma
        self.lmbda=lmbda
        self.policy=Policy(input_dim,hidden_dim,action_space).to(self.device)
        self.critic=CriticNet(input_dim,hidden_dim).to(self.device)
        self.criticOptimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

        self.device=device

    def PolicyLearn(self,states,actions,old_log_probs,old_action_dists,advantage):
        

    def update(self,states,actions,rewards,next_states,dones):
        states=torch.tensor(states,dtype=torch.float32).to(self.device)
        actions=torch.tensor(actions).view(-1,1).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(next_states,dtype=torch.float32).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float32).view(-1,1).to(self.device)

        TD_target=rewards+self.gamma*(1-dones)*self.critic(next_states)
        TD_delta=TD_target-self.critic(states)
        advantage=compuate_GAE(self.lmbda,self.gamma,TD_delta.detach())
        old_log_probs=torch.log(self.policy(states).gather(1,actions)).detach()
        old_action_dists=torch.distributions.Categorical(self.policy(states)).detach()
        TD_loss=torch.mean(F.mse_loss(self.critic(states),TD_target.detach()))
        self.criticOptimizer.zero_grad()
        TD_loss.backward()
        self.criticOptimizer.step()

        self.PolicyLearn(states,actions,old_log_probs,old_action_dists,advantage)
