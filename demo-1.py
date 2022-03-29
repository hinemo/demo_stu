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
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  
    
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
    optimizer.zero_grad()  #梯度清零，也就是将Loss关于weight的导数置为0，如果不进行这一步操作，那么每一次epoch都会使得梯度叠加，造成梯度爆炸。
    outputs = model(inputs) #推导
    loss = criterion(outputs, targets) #求解loss,计算损失函数
    loss.backward()  #反向传播求解梯度，来源是torch.autograd.backward，使用backward()可以计算loss的梯度，代码loss.backward()执行后，loss中各个parameters的梯度得到更新，比如loss的parameter中有个变量是x，那么执行loss.backward之后，loss关于变量x的梯度就保存在x.grad中，可以通过x.grad来查看。
    optimizer.step() # 更新权重参数， 只对loss的parameter做单步更新，所谓的更新，就是新parameter值=原parameter值−learningr​ate∗parameter.grad(导数值)
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
