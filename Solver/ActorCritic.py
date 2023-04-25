import torch
import torch.nn.functional as F
import numpy as np

class QNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(QNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class VNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(VNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class BaseActorCritic:
    def __init__(self,gamma,learning_rate,param_list,device):
        self.gamma=gamma
        self.param_list=param_list
        self.device=device

        self.qnet=QNet(param_list[0],param_list[1],param_list[2]).to(self.device)
        self.vnet=VNet(param_list[0],param_list[1]).to(self.device)

        self.qOptimizer=torch.optim.Adam(self.qnet.parameters(),lr=learning_rate)
        self.vOptimizer=torch.optim.Adam(self.vnet.parameters(),lr=learning_rate)

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float32).to(self.device)
        prob=self.qnet(state)
        prob_dist=torch.distributions.Categorical(probs=prob)
        action=prob_dist.sample()
        return action.item()

    def update(self,transition):
        states=torch.tensor(transition['states'],dtype=torch.float32).to(self.device)
        actions=torch.tensor(transition['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition['next_states'],dtype=torch.float32).to(self.device)
        dones=torch.tensor(transition['dones'],dtype=torch.float32).view(-1,1).to(self.device)

        self.qOptimizer.zero_grad()
        self.vOptimizer.zero_grad()
        td_target=rewards+self.gamma*self.vnet(next_states)*(1-dones)
        td_values=self.vnet(states)
        td_error=td_target-self.vnet(states)
        log_probs=torch.log(self.qnet(states).gather(1,actions))
        qLoss=torch.mean(-log_probs*td_error.detach())
        qLoss.backward()
        vLoss=torch.mean(F.mse_loss(td_target.detach(),td_values))
        vLoss.backward()
        self.qOptimizer.step()
        self.vOptimizer.step()


