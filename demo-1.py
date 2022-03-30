import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


# Hyper Parameters
input_size = 1
output_size = 1
num_epochs = 1000
learning_rate = 0.001

x_train = np.array([[2.3], [4.4], [3.7], [6.1], [7.3], [2.1],[5.6], [7.7], [8.7], [4.1],

                    [6.7], [6.1], [7.5], [2.1], [7.2],

                    [5.6], [5.7], [7.7], [3.1]], dtype=np.float32)

#xtrain生成矩阵数据

y_train = np.array([[3.7], [4.76], [4.], [7.1], [8.6], [3.5],[5.4], [7.6], [7.9], [5.3],

                    [7.3], [7.5], [8.5], [3.2], [8.7],

                    [6.4], [6.6], [7.9], [5.3]], dtype=np.float32)
plt.figure() 
#画图散点图
plt.scatter(x_train,y_train)
plt.xlabel('x_train')
#x轴名称
plt.ylabel('y_train')
#y轴名称
#显示图片
#plt.show()


# Linear Regression Model
class LinearRegression(nn.Module):
    # ：前一层网络神经元的个数
    def __init__(self, input_size, output_size,bias=True):
        super(LinearRegression, self).__init__()
        #in_features指的是输入的二维张量的大小,out_features指的是输出的二维张量的大小
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)  
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size)

# Loss and Optimizer 
#均方损失函数 loss(xi,yi)=(xi−yi)2
criterion = nn.MSELoss()
#第一个参数包括权重w，和偏置b等是神经网络中的参数，也是SGD优化的重点;第二个参数lr是学习率;
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the Model 
for epoch in range(num_epochs):
    #Variable(tensor, requires_grad = True),tensor不能反向传播，variable可以反向传播,tensor变成variable之后才能进行反向传播
  
    inputs = Variable(torch.from_numpy(x_train),requires_grad = True)
    targets = Variable(torch.from_numpy(y_train),requires_grad = True)
   
    # Forward + Backward + Optimize
    optimizer.zero_grad()  #梯度清零
    outputs = model(inputs) #推导
    loss = criterion(outputs, targets) #求解loss,计算损失函数
    loss.backward()  #反向传播求解梯度
    optimizer.step() # 更新权重参数    
    if (epoch+1) % 5 == 0:#每5次epoch打印一次结果
        print ('Epoch [%d/%d], Loss: %.4f' 
               %(epoch+1, num_epochs, loss.item()))
        
# Plot the graph
model.eval()
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, predicted, label='predict')
plt.legend()
plt.show()