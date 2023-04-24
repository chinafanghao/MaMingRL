import sys
if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay
import torch
import torch.nn.functional as F
import numpy as np

class Qnet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self,env,epsilon,gamma,learning_rate,param_list,target_update,device):
        self.env=env
        self.epsilon=epsilon
        self.gamma=gamma
        self.device=device
        self.param_list=param_list
        self.target_update=target_update

        self.Qnet=Qnet(param_list[0],param_list[1],param_list[2]).to(device)
        self.Qnet_target=Qnet(param_list[0],param_list[1],param_list[2]).to(device)
        self.optimizer=torch.optim.Adam(self.Qnet.parameters(),lr=learning_rate)

        self.counts=0

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.param_list[-1])
        else:
            state=torch.tensor(np.array([state]),dtype=torch.float).to(self.device)
            action=self.Qnet(state).argmax().item()
        return action

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        truncateds=torch.tensor(transition_dict['truncateds'],dtype=torch.float).view(-1,1).to(self.device)

        q_value=self.Qnet(states).gather(1,actions)
        max_next_q_value=self.Qnet_target(next_states).max(1)[0].view(-1,1)
        q_targets=rewards+self.gamma*max_next_q_value*(1-dones)+100*truncateds
        dqn_loss=torch.mean(F.mse_loss(q_value,q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.counts%self.target_update==0:
            self.Qnet_target.load_state_dict(self.Qnet.state_dict())
        self.counts+=1






        