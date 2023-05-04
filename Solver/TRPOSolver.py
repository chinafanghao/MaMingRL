import sys

if "../" not in sys.path:
  sys.path.append("../")

import torch
import torch.nn.functional as F
import numpy as np
from utils.util import compuate_GAE
import copy

class CriticNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(CriticNet,self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class PolicyNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(PolicyNet, self).__init__()
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class TRPO:
    def __init__(self,gamma,lmbda,input_dim,hidden_dim,action_space,lr,alpha,kl_constraint,device):
        self.gamma=gamma
        self.lmbda=lmbda
        self.alpha=alpha
        self.kl_constraint=kl_constraint
        self.policy=PolicyNet(input_dim,hidden_dim,action_space).to(device)
        self.critic=CriticNet(input_dim,hidden_dim).to(device)
        self.criticOptimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

        self.device=device

    def take_action(self,state):
        state=torch.tensor(np.array([state]),dtype=torch.float32).to(self.device)
        probs=self.policy(state)
        action_dist=torch.distributions.Categorical(probs=probs)
        action=action_dist.sample()
        return action.item()

    def compute_obj(self,states,actions,old_log_probs,advantage,actor):
        log_probs=torch.log(actor(states).gather(1,actions))
        ratio=torch.exp(log_probs-old_log_probs)
        return torch.mean(ratio*advantage)

    def Hessian_matrix_vector_product(self,states,old_action_dists,vector):
        action_dists=torch.distributions.Categorical(self.policy(states))
        kl=torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,action_dists))
        kl_grad=torch.autograd.grad(kl,self.policy.parameters(),create_graph=True)
        kl_grad_vector=torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product=torch.dot(kl_grad_vector,vector)
        grad2=torch.autograd.grad(kl_grad_vector_product,self.policy.parameters())
        grad2_vector=torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self,grad,states,old_action_dists):
        x=torch.zeros_like(grad)
        r=grad.clone()
        p=grad.clone()
        rdotr=torch.dot(r,r)
        for i in range(10):
            Hp=self.Hessian_matrix_vector_product(states,old_action_dists,p)
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

    def line_search(self,states,actions,advantage,old_log_probs,old_action_dists,max_vec):
        old_para=torch.nn.utils.convert_parameters.parameters_to_vector(self.policy.parameters())
        old_obj=self.compute_obj(states,actions,old_log_probs,advantage,self.policy)
        for i in range(15):
            coef=self.alpha**i
            new_para=old_para+coef*max_vec
            new_policy=copy.deepcopy(self.policy)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para,new_policy.parameters())
            new_action_dists=torch.distributions.Categorical(new_policy(states))
            kl_div=torch.mean(torch.distributions.kl.kl_divergence(old_action_dists,new_action_dists))
            new_obj=self.compute_obj(states,actions,old_log_probs,advantage,new_policy)
            if new_obj>old_obj and kl_div<self.kl_constraint:
                return new_para
        return old_para

    def PolicyLearn(self,states,actions,old_log_probs,old_action_dists,advantage):
        origin_obj=self.compute_obj(states,actions,old_log_probs,advantage,self.policy)
        grads=torch.autograd.grad(origin_obj,self.policy.parameters())
        obj_grad=torch.cat([grad.view(-1) for grad in grads]).detach()

        descent_direction=self.conjugate_gradient(obj_grad,states,old_action_dists)

        Hd=self.Hessian_matrix_vector_product(states,old_action_dists,descent_direction)
        max_coef=torch.sqrt(2*self.kl_constraint/(torch.dot(descent_direction,Hd)+1e-8))
        new_para=self.line_search(states,actions,advantage,old_log_probs,old_action_dists,descent_direction*max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para,self.policy.parameters())


    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float32).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float32).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float32).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float32).view(-1,1).to(self.device)

        TD_target=rewards+self.gamma*(1-dones)*self.critic(next_states)
        TD_delta=TD_target-self.critic(states)

        advantage=compuate_GAE(self.lmbda,self.gamma,TD_delta.cpu()).to(self.device)
        old_log_probs=torch.log(self.policy(states).gather(1,actions)).detach()
        old_action_dists=torch.distributions.Categorical(self.policy(states).detach())

        TD_loss=torch.mean(F.mse_loss(self.critic(states),TD_target.detach()))
        self.criticOptimizer.zero_grad()
        TD_loss.backward()
        self.criticOptimizer.step()

        self.PolicyLearn(states,actions,old_log_probs,old_action_dists,advantage)
