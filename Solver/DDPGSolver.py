import torch
import torch.nn.functional as F
import numpy as np

class TwoLayerFC(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,activation=F.relu,out_fn=lambda x:x):
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc3=torch.nn.Linear(hidden_dim,output_dim)

        self.activation=activation
        self.out_fn=out_fn

    def forward(self,x):
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        return self.out_fn(self.fc3(x))

class DDPG:
    def __init__(self,gamma,tau,sigma,input_dim,hidden_dim,output_dim,actor_lr,critic_lr,discrete,action_bound,device):
        out_fn=(lambda x:x) if discrete else (lambda x:torch.tanh(x)*action_bound)
        self.actor=TwoLayerFC(input_dim,hidden_dim,output_dim,activation=F.relu,out_fn=out_fn).to(device)
        self.actor_target=TwoLayerFC(input_dim,hidden_dim,output_dim,activation=F.relu,out_fn=out_fn).to(device)
        self.critic=TwoLayerFC(input_dim,hidden_dim,1,activation=F.relu,out_fn=out_fn).to(device)
        self.critic_target = TwoLayerFC(input_dim, hidden_dim, 1, activation=F.relu, out_fn=out_fn).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actorOptimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.criticOptimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)

        self.gamma=gamma
        self.device=device
        self.tau=tau
        self.sigma=sigma
        self.action_dim=output_dim
        self.discrete=discrete

    def take_action(self,state):
        state=torch.tensor(np.array(state),dtype=torch.float32).to(self.device)
        action=self.actor(state).item()
        action+=self.sigma*np.random.randn(self.action_dim)
        return action

    def soft_update(self,net,target_net):
        for net_para,target_para in zip(net.parameters(),target_net.parameters()):
            target_para.data.copy_(target_para*(1-self.tau)+net_para*self.tau)

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float32).to(self.device)
        if self.discrete:
            actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        else:
            actions=torch.tensor(transition_dict['actions'],dtype=torch.float32).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float32).to(self.device)

        next_q_value=self.critic_target(torch.cat([next_states,self.actor_target(next_states)],dim=1))
        q_target=rewards+self.gamma*next_q_value*(1-dones)

        critic_loss=torch.mean(F.mse_loss(self.critic(torch.cat([states,actions],dim=1)),q_target))
        self.criticOptimizer.zero_grad()
        critic_loss.backward()
        self.criticOptimizer.step()

        actor_loss=-torch.mean(self.critic(torch.cat([states,actions],dim=1)))
        self.actorOptimizer.zero_grad()
        actor_loss.backward()
        self.actorOptimizer.step()

        self.soft_update(self.actor,self.actor_target)
        self.soft_update(self.critic,self.critic_target)
