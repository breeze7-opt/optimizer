import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import time
from ResNet import *
from VGG16 import *
from yogi import Yogi
from AdaBound import AdaBound
from adamod import AdaMod
from Adan import Adan

batch_size = 64
'''datasets'''
train_data = torchvision.datasets.CIFAR100(root="F/CIFAR100", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR100(root="F/CIFAR100", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

'''load model'''
model =ResNet34().cuda()

'''optimizer'''
from AdaGC import AdaGC
optimizer =AdaGC(model.parameters())
print(optimizer)

'''loss'''
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()
start = time.time()

train_loss = []
train_accs = []
test_loss = []
test_accs =[]

'''train'''
def train(epoch):
    print('           第 {} 轮'.format(epoch))
    model.train()
    total_train_loss= 0
    total_train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()   #清空过往梯度
        output = model(data)
        loss = criterion(output,target)
        total_train_loss +=loss.item()
        train_acc = (output.argmax(1) == target).sum()
        total_train_acc +=  train_acc
        loss.backward()    #反向传播
        optimizer.step()   #根据梯度更新网络参数
    train_loss.append(total_train_loss)
    train_accs.append((100.* int(total_train_acc.item()) / train_data_size))

    print("训练集上的Loss：{}".format(total_train_loss))
    print("训练集上的准确率：{}%".format(int(total_train_acc.item()) * 100 / train_data_size))

def test():
    model.eval()
    total_test_loss = 0
    total_test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data),Variable(target)
            if torch.cuda.is_available():
              data = data.cuda()
              target = target.cuda()
            output = model(data)
            total_test_loss += criterion(output, target).item()
            test_acc = (output.argmax(1) == target).sum()
            total_test_acc += test_acc


    test_loss.append(total_test_loss)
    test_accs.append(100.* int(total_test_acc.item())/test_data_size)

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}%".format(int(total_test_acc.item()) * 100 / test_data_size))

for epoch in range(1, 101):
    train(epoch)
    test()
end = time.time()

print('total train loss',train_loss)
print('total test loss',test_loss)
print('total train acc',train_accs)
print('total test acc',test_accs)
print('最大精度： {:.2f}%'.format(max(test_accs)))
