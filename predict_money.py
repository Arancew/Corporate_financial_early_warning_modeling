# 5，公司财务预警建模
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 获取数据
filepath = '.\data.mat'
dict_data = scio.loadmat(filepath)
input_train = dict_data['input_train']
output_train = dict_data['output_train']
input_test = dict_data['input_test']
output_test = dict_data['output_test']

output_train = np.maximum(output_train, 0)  # 转换成0，1
output_train = output_train.astype(int)
output_test = np.maximum(output_test, 0)
output_test = output_test.astype(int)


# 将数据转换成张量形式
def to_tensor(x):
    x = pd.DataFrame(x)
    x = x.stack()
    x = x.unstack(0)
    x = torch.Tensor(x.values)
    return x


tmp = torch.tensor([10, 10000, 100, 10, 10, 100000, 100, 10, 10, 100])
tmp = tmp.type(torch.float)
input_train = to_tensor(input_train)
output_train = to_tensor(output_train).resize(1, 1057)[0]
input_test = to_tensor(input_test)
output_test = to_tensor(output_test).resize(1, 350)[0]
# 归一化
input_train = input_train.div_(tmp)
input_test = input_test.div_(tmp)
output_train = output_train.type(torch.long)
output_test = output_test.type(torch.long)


#  定义神经网络
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


# 定义神经网络和优化器，还有损失函数
net = mlp()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(input_train, output_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=5,
    shuffle=True,
)
cnt = 0
yy = []
for epoch in range(20):
    for step, (x, y) in enumerate(loader):
        output = net(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            # print(output)
            test_output = net(input_test)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == output_test) / output_test.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
            yy.append(accuracy)
            cnt += 1

print('最终结果')
test_output = net(input_test)
pred_y = torch.max(test_output, 1)[1].data.squeeze()
accuracy = sum(pred_y == output_test) / output_test.size(0)
print('train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
plt.plot(yy,color='red')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
print(pred_y)
