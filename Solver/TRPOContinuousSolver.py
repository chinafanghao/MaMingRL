import sys

if "../" not in sys.path:
  sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
from utils.util import compuate_GAE
import copy

class PolicyNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PolicyNet,self).__init__()
        self.fc=torch.nn.Linear(input_dim,hidden_dim)
        self.fc_mu=torch.nn.Linear(hidden_dim,output_dim)
        self.fc_std=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc(x))
        mu=2*F.tanh(self.fc_mu(x))
        std=F.softplus(self.fc_std(x))
        return mu,std

class ValueNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(ValueNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class TRPOContinuous:
    def __init__(self,gamma,lr,lmbda,alpha,input_dim,hidden_dim,output_dim,kl_constraints,device):
        self.gamma=gamma
        self.lmbda=lmbda
        self.device=device
        self.alpha=alpha
        self.kl_constraints=kl_constraints

        self.actor=PolicyNet(input_dim,hidden_dim,output_dim).to(device)
        self.critic=ValueNet(input_dim,hidden_dim).to(device)
        self.criticOptimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

    def take_action(self,state):
        state=torch.tensor([state],dtype=torch.float32).to(self.device)
        mu,std=self.actor(state)
        action_dist=torch.distributions.Normal(mu,std)
        action=action_dist.sample()
        return [action.item()]

    def compute_obj(self,states,actions,old_log_probs,advantage,actor):
        mu,std=actor(states)
        new_action_dists=torch.distributions.Normal(mu,std)
        new_log_probs=new_action_dists.log_prob(actions)
        ratio=torch.exp(new_log_probs-old_log_probs)
        return torch.mean(ratio*advantage)

    def Hessian_matrix_vector_product(self,states,vector,old_action_dists,damping=0.1):
        mu,std=self.actor(states)
        new_action_dists=torch.distributions.Normal(mu,std)
        kl=torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,new_action_dists))
        kl_grad=torch.autograd.grad(kl,self.actor.parameters(),create_graph=True)
        kl_grad_vector=torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product=torch.dot(kl_grad_vector,vector)
        grad2=torch.autograd.grad(kl_grad_vector_product,self.actor.parameters())
        grad2_vector=torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector+damping*vector

    def conjugate_gradient(self,grads,states,old_action_dists):
        x=torch.zeros_like(grads)
        r=grads.clone()
        p=grads.clone()
        rdotr=torch.dot(r,r)
        for i in range(10):
            Hp=self.Hessian_matrix_vector_product(states,p,old_action_dists)
            alpha=rdotr/torch.dot(p,Hp)
            x+=alpha*p
            r-=alpha*Hp
            new_rdotr=torch.dot(r,r)
            if new_rdotr<1e-10:
                break
            beta=new_rdotr/rdotr
            p=r+beta*p
            rdotr=new_rdotr
        return x

    def line_search(self,states,actions,old_log_probs,old_action_dists,advantage,max_vector):
        old_para=torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj=self.compute_obj(states,actions,old_log_probs,advantage,self.actor)
        for i in range(15):
            alpha=self.alpha**i
            new_para=old_para+alpha*max_vector
            new_actor=copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para,new_actor.parameters())
            new_mu,new_std=new_actor(states)
            new_action_dists=torch.distributions.Normal(new_mu,new_std)
            kl_div=torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,new_action_dists))
            new_obj=self.compute_obj(states,actions,old_log_probs,advantage,new_actor)

            if new_obj>old_obj and kl_div<self.kl_constraints:
                return new_para
        return old_para

    def PolicyLearn(self,states,actions,old_action_dists,old_log_probs,advantage):
        surrogate_obj=self.compute_obj(states,actions,old_log_probs,advantage,self.actor)
        grads=torch.autograd.grad(surrogate_obj,self.actor.parameters())
        obj_grad=torch.cat([grad.view(-1) for grad in grads]).detach()

        descent_direction=self.conjugate_gradient(obj_grad,states,old_action_dists)
        Hx=self.Hessian_matrix_vector_product(states,descent_direction,old_action_dists)
        max_vector=torch.sqrt(2*self.kl_constraints/(torch.dot(descent_direction,Hx)+1e-8))
        max_vector=max_vector*descent_direction

        new_para=self.line_search(states,actions,old_log_probs,old_action_dists,advantage,max_vector)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para,self.actor.parameters())

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float32).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float32).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float32).view(-1,1).to(self.device)

        rewards=(rewards+8.0)/8.0
        TD_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        TD_delta=TD_target-self.critic(states)
        advantage=compuate_GAE(self.lmbda,self.gamma,TD_delta.cpu()).to(self.device)
        mu,std=self.actor(states)
        old_action_dists=torch.distributions.Normal(mu.detach(),std.detach())
        old_log_probs = old_action_dists.log_prob(actions)

        td_loss=torch.mean(F.mse_loss(self.critic(states),TD_target.detach()))
        self.criticOptimizer.zero_grad()
        td_loss.backward()
        self.criticOptimizer.step()
        self.PolicyLearn(states,actions,old_action_dists,old_log_probs,advantage)