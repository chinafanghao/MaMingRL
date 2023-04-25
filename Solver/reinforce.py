import numpy as np
import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PolicyNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))
class ValueNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(ValueNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class REINFORCE:
    def __init__(self,gamma,learning_rate,param_list,device):
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.param_list=param_list
        self.device=device

        self.qnet=PolicyNet(param_list[0],param_list[1],param_list[2]).to(self.device)
        self.optimizer=torch.optim.Adam(self.qnet.parameters(),lr=learning_rate)

    def take_action(self,state):
        state=torch.tensor(np.array([state]),dtype=torch.float32).to(self.device)
        prob=self.qnet(state)
        prob_dist=torch.distributions.Categorical(probs=prob)
        action=prob_dist.sample()
        return action.item()

    def update(self,transition):
        states_list=transition['states']
        actions_list=transition['actions']
        rewards_list=transition['rewards']

        G=0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rewards_list))):
            reward=rewards_list[i] #torch.tensor([rewards_list[i]],dtype=torch.float32).view(-1,1).to(self.device)
            G=self.gamma*G+reward
            state = torch.tensor([states_list[i]], dtype=torch.float32).to(self.device)
            action = torch.tensor([actions_list[i]]).view(-1, 1).to(self.device)
            loss=-torch.log(self.qnet(state).gather(1,action))*G
            loss.backward()
        self.optimizer.step()

class REINFOCEWithBaseline:
    def __init__(self,gamma,learning_rate,param_list,device):
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.device=device

        self.qnet=PolicyNet(param_list[0],param_list[1],param_list[2]).to(self.device)
        self.vnet=ValueNet(param_list[0],param_list[1]).to(self.device)

        self.qOptimizer=torch.optim.Adam(self.qnet.parameters(),lr=learning_rate)
        self.vOptimizer=torch.optim.Adam(self.vnet.parameters(),lr=learning_rate)

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float32).to(self.device)
        prob=self.qnet(state)
        prob_dist=torch.distributions.Categorical(probs=prob)
        action=prob_dist.sample()
        return action.item()

    def update(self,transition):
        states_list=transition['states']
        actions_list=transition['actions']
        rewards_list=transition['rewards']

        G=0
        self.qOptimizer.zero_grad()
        self.vOptimizer.zero_grad()
        vloss=0
        for i in reversed(range(len(rewards_list))):
            reward=rewards_list[i] #torch.tensor([rewards_list[i]],dtype=torch.float32).view(-1,1).to(self.device)
            state=torch.tensor([states_list[i]],dtype=torch.float32).to(self.device)
            action=torch.tensor([actions_list[i]]).view(-1,1).to(self.device)

            G=self.gamma*G+reward
            logP=torch.log(self.qnet(state).gather(1,action))
            ploss=-logP*(G-self.vnet(state).detach())
            ploss.backward()
            vloss+=F.mse_loss(torch.tensor([G],dtype=torch.float32).view(-1,1).to(self.device),self.vnet(state))
        vloss.backward()
        self.qOptimizer.step()
        self.vOptimizer.step()
