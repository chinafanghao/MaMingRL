import torch
import torch.nn.functional as F
import numpy as np

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.mu=torch.nn.Linear(hidden_dim,output_dim)
        self.std=torch.nn.Linear(hidden_dim,output_dim)
        self.action_bound=action_bound

    def forward(self,x):
        x=F.relu(self.fc1(x))
        mu=self.mu(x)
        std=F.softplus(self.std(x))
        action_dist=torch.distributions.Normal(mu,std)
        normal_sample=action_dist.rsample()
        log_prob=action_dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        #log_prob=log_prob-torch.log(1-torch.pow(action,2)+1e-7)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action=action*self.action_bound
        return action,log_prob

class QValueNetContinuous(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1=torch.nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc_out=torch.nn.Linear(hidden_dim,1)

    def forward(self,state,action):
        cat=torch.cat([state,action],dim=1)
        x=F.relu(self.fc1(cat))
        x=F.relu(self.fc2(x))
        return self.fc_out(x)

class SACContinuous:
    def __init__(self,gamma,tau,state_dim,action_dim,hidden_dim,actor_lr,critic_lr,alpha_lr,target_entropy,action_bound,device):
        self.gamma=gamma
        self.tau=tau
        self.target_entropy=target_entropy
        self.device=device

        self.actor=PolicyNetContinuous(state_dim,hidden_dim,action_dim,action_bound).to(self.device)

        self.critic1=QValueNetContinuous(state_dim,hidden_dim,action_dim).to(self.device)
        self.critic2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic1_target = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic2_target = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actorOptimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic1Optimizer=torch.optim.Adam(self.critic1.parameters(),lr=critic_lr)
        self.critic2Optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha=torch.tensor(np.log(0.01),dtype=torch.float32).to(self.device)
        self.log_alpha.requires_grad=True
        self.log_alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=alpha_lr)

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float32).to(self.device)
        action=self.actor(state)[0]
        return [action.item()]

    def Qtarget(self,rewards,next_states,dones):
        next_actions,log_prob=self.actor(next_states)
        entropy=-log_prob
        q1_value=self.critic1_target(next_states,next_actions)
        q2_value=self.critic2_target(next_states,next_actions)
        next_value=torch.min(q1_value,q2_value)+self.log_alpha.exp()*entropy
        td_targets=rewards+self.gamma*next_value*(1-dones)
        return td_targets

    def soft_update(self,net,target_net):
        for net_para,target_net_para in zip(net.parameters(),target_net.parameters()):
            target_net_para.data.copy_((1-self.tau)*target_net_para.data+self.tau*net_para.data)

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float32).to(self.device)
        actions=torch.tensor(transition_dict['actions'],dtype=torch.float32).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float32).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float32).view(-1,1).to(self.device)

        rewards=(rewards+8.0)/8.0
        td_target=self.Qtarget(rewards,next_states,dones)
        critic1_loss=torch.mean(F.mse_loss(self.critic1(states,actions),td_target.detach()))
        critic2_loss=torch.mean(F.mse_loss(self.critic2(states,actions),td_target.detach()))
        self.critic1Optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1Optimizer.step()
        self.critic2Optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2Optimizer.step()

        new_actions,log_probs=self.actor(states)
        entropy=-log_probs
        q1_value=self.critic1(states,new_actions)
        q2_value=self.critic2(states,new_actions)
        actor_loss=torch.mean(-self.log_alpha.exp()*entropy-torch.min(q1_value,q2_value))
        self.actorOptimizer.zero_grad()
        actor_loss.backward()
        self.actorOptimizer.step()

        alpha_loss=torch.mean((entropy-self.target_entropy).detach()*self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1,self.critic1_target)
        self.soft_update(self.critic2,self.critic2_target)


