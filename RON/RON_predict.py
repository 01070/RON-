import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch
import xlrd #读取excel的库
import matplotlib.pyplot as plt
plt.figure()
fig, ax = plt.subplots()
resArray=[] #先声明一个空list
avg_arr = []
dataset = []
data = xlrd.open_workbook("train.xlsx") #读取文件
table = data.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table.nrows): #table.nrows表示总行数
    line=table.row_values(i) #读取每行数据，保存在line里面，line是list
    resArray.append(line) #将line加入到resArray中，resArray是二维list
resArray=np.array(resArray) #将resArray从二维list变成数组
X = resArray[:, 1:]
y = resArray[:, 0]

X_min = np.min(X, axis=0)
X_min_range = [85.3, 0, 10, 5500, 0, 50, 100, 0, 0.1, 15, 700, 0, 0, 0, 4, 400, 0, 0, 5, 1.2, 0, -125, 50]
X_max = np.max(X, axis=0)
X_max_range =[91.7, 3500, 1500, 9000, 0.25, 150, 200, 30, 0.2, 45, 25000, 400, 3.5, 120, 60, 450, 300, 400, 150, 3.6, 12000, 0.5, 110]
X_ = (X - X_min_range) / (X_max - X_min_range)

y_min = np.min(y)
y_max = np.max(y)
y_ = (y - y_min) / (y_max - y_min)

X_tensor = torch.FloatTensor(X_)
y_tensor = torch.FloatTensor(y)

for i in range(325):
    dataset.append((X_tensor[i], y_tensor[i]))

train_data = DataLoader(dataset[:300], 32, True, drop_last=True)
test_data = DataLoader(dataset[300:], 25, True, drop_last=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin = nn.Linear(23, 1)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.lin(x)
        # x = self.tanh(x)
        # x = self.lin2(x)
        return x

model = Model()
train_loss_list = []
test_loss_list = []
def test():
    model.eval()
    with torch.no_grad():
        for batch, test_target in test_data:
            test_predict = model(batch)
            loss = criterion(test_predict, test_target)
    return loss.item()
# model.load_state_dict(torch.load('./weight'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
for epoch in range(50):
    for batch, target in train_data:
        predict = model(batch)
        optimizer.zero_grad()
        loss = criterion(predict, target)
        loss.backward()
        optimizer.step()
    train_loss_list.append(loss.item())
    test_loss_list.append(test())

T=np.arange(50)
plt.plot(T, train_loss_list, color='r',label='train')
plt.plot(T, test_loss_list, color='b',label='test')
plt.plot()
plt.grid(False)
plt.legend(loc='best',frameon=False,fontsize='small' )
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()
print('the minimum test loss:{:.3f}'.format(min(test_loss_list)))

