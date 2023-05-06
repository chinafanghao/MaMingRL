import torch
import torch.nn.functional as F
import numpy as np

class QValueNet(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        self.fc1=torch.nn.Linear(input_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,output_dim)
