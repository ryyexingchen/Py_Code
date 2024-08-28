import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import functional as f
import lib
import os
class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        self.Linear1 = nn.Linear(lib.max_word_len,256)
        self.Linear3 = nn.Linear(256, 32)
        self.Linear4 = nn.Linear(32,1)

    def forward(self, input):
        x = self.Linear1(input)
        x = f.relu(x)
        # # x = self.Linear2(x)
        # # x = f.relu(x)
        x = self.Linear3(x)
        x = f.relu(x)
        out = self.Linear4(x)
        return out

my_linear = MyLinear()
optimizer = SGD(my_linear.parameters(), lib.learning_rate)
loss_fn = nn.MSELoss()
if os.path.exists(lib.model_path):
    my_linear.load_state_dict(torch.load(lib.model_path))
    optimizer.load_state_dict(torch.load(lib.optimizer_path))
def linear_train(data,epoch):
    # data的形状: [[wordVec,ans]*len(data)]
    for j in range(epoch):
        for i in range(len(data)):
            input = torch.Tensor([data[i][0]])
            label = torch.Tensor([data[i][1]])
            predict = my_linear(input)
            loss = loss_fn(predict,label)
            print(j,i,label,predict.item(),loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 200 == 0:
                torch.save(my_linear.state_dict(), lib.model_path)
                torch.save(optimizer.state_dict(), lib.optimizer_path)
    torch.save(my_linear.state_dict(), lib.model_path)
    torch.save(optimizer.state_dict(), lib.optimizer_path)
