import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#model
class Linear_Qnet(nn.Module): 
    
    def __init__(self, input_size , hidden_size , ouput_size):
        super().__init__()
        
        # self.liner1 = nn.Linear(input_size, hidden_size) #first layer with (11,128)
        # self.liner2 = nn.Linear(hidden_size, ouput_size) #second layer with (128,3) output will be left, right, straight
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, ouput_size)
    
    def forward(self, x): #act like model.predict it will give Q-value 
        # x = F.relu(self.liner1(x))
        # x= self.liner2(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
    def save(self, file_name="model.pth"):
        model_folder = "./model"
        
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            
        file_name = os.path.join(model_folder, file_name)
        
        torch.save(self.state_dict(), file_name); #save model like done using pickel

#training
class QTrainer:
    
    def __init__(self, model, lr, gamma):
        self.lr = lr;
        self.gamma  = gamma;
        self.model = model
        self.optimizer =  optim.Adam(model.parameters(), lr=self.lr);
        self.criterion =  nn.MSELoss()  #loss function mean square error
        
    def train_step(self, state, action, reward, next_state, done):
       
       state = torch.tensor(state, dtype=torch.float)
       action = torch.tensor(action, dtype=torch.float)
       reward = torch.tensor(reward, dtype=torch.float)
       next_state = torch.tensor(next_state, dtype=torch.float)
       
       
       if len(state.shape) == 1: #check if there is 1D array convert it into 2D because Qnet accept only 2D aarray
           state =  torch.unsqueeze(state,0)
           action =  torch.unsqueeze(action,0)
           reward =  torch.unsqueeze(reward,0)
           next_state =  torch.unsqueeze(next_state,0)
           done  = (done,)
           
           #Bellman equation 
           #1.predict q value with current state 
           # Q =  model.predict(state0)
           
           predicted  = self.model(state)
           
           target = predicted.clone()
           
           #2. r +y * max(next predicted q value) -> only do this if game is not over
           #if game over Q_new = reward
           #else Qnew = reward + gamma * max(next_state)
           # Qnew = r+y*max(predicted)
           for idx in range(len(done)):
                    Q_new = reward[idx]
                    if not done[idx]:
                        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                        
                    target[idx][torch.argmax(action[idx]).item()] = Q_new 
                         
           #loss function
           self.optimizer.zero_grad()
           loss = self.criterion(target,predicted) # (y_true , ypred)
           loss.backward()
           
           self.optimizer.step()