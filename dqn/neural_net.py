import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Net(nn.Module):
    def __init__(self,seed):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)  
        self.fc2 = nn.Linear(10, 5)  
        self.fc3 = nn.Linear(5, 1) 
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self,input_ls,label_ls):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for i in range(len(input_ls)):
            inputs = torch.tensor(input_ls[i],dtype=torch.float).unsqueeze(0)
            labels = torch.tensor(label_ls[i],dtype=torch.float).unsqueeze(0)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
class Q_network:
    def __init__(self,max_cap=1000):
        seed = random.randint(0, 1000)
        self.target_network = self.build_model(seed)
        self.online_network = self.build_model(seed)
        self.replay_memory = deque(max_memory=max_cap)
    def build_model(seed):
        model = Net(seed=seed)
        return model
    
