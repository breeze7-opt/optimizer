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
from adabelief import AdaBelief
from yogi import Yogi
from AdaBound import AdaBound
from adamod import AdaMod
from Adan import Adan
from AdaMG import AdaMG
#from aggmo import AggMo
#from lamb import *

batch_size = 64
'''设置数据集'''
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

'''建立网络'''


#定义网络
model =ResNet34().cuda()

'''挑选优化器'''
from new_5 import new_5
#optimizer =AdaBound(model.parameters(), lr= 0.001)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
print(optimizer)

'''损失函数'''
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    criterion = criterion.cuda()
start = time.time()

train_loss = []
train_accs = []
test_loss = []
test_accs =[]
time_train = []
time_test = []
target_accuracy =54 #85.5 #87

'''开始训练'''
def train(epoch):
    print('           第 {} 轮'.format(epoch))
    train_time_1 = time.time()
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

    train_time_2 = time.time()
    time_train.append(train_time_2-train_time_1)

    print("训练集上的Loss：{}".format(total_train_loss))
    print("训练集上的准确率：{}%".format(int(total_train_acc.item()) * 100 / train_data_size))
    print('该轮训练时间：{}s'.format(train_time_2-train_time_1))
        #if batch_idx % 50 == 0:
         #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
         #              100. * batch_idx / len(train_loader), loss.item()))
def test():
    model.eval()
    total_test_loss = 0
    total_test_acc = 0
    test_time_1 = time.time()
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

    test_time_2 = time.time()
    test_loss.append(total_test_loss)
    test_accs.append(100.* int(total_test_acc.item())/test_data_size)
    time_test.append(test_time_2-test_time_1)

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}%".format(int(total_test_acc.item()) * 100 / test_data_size))
    print('该轮测试时间：{}s'.format(test_time_2-test_time_1))
    '''
    if (int(total_test_acc.item()) * 100 / test_data_size) >= target_accuracy:
        use_time = time.time()
        time_taken = use_time - start
        print("达到精度 {}% 所用的时间：{} 秒".format(target_accuracy, time_taken))
        break
    '''
    #print('           第 {} 轮'.format(epoch))
    #print('Test set: Average loss: {:.4f}'.format(test_loss))
    #print('Accuracy: {}/{} ({:.2f}%)'.format(correct, test_data_size,100. * correct / test_data_size))
for epoch in range(1, 201):
    train(epoch)
    test()
    # 判断是否达到目标准确率，达到则跳出循环
    if  test_accs[epoch-1] >= target_accuracy:
        use_time = time.time()
        time_taken = use_time - start
        print("达到精度 {}% 所用的时间：{} 秒".format(target_accuracy, time_taken))
        break
end = time.time()
print('total train loss',train_loss)
print('total test loss',test_loss)
print('total train acc',train_accs)
print('total test acc',test_accs)
#print('所用时间为：',end-start,'s')
print('最大精度： {:.2f}%'.format(max(test_accs)))
print('train time list', time_train)
print('test time list', time_test)