import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from AdaBound import AdaBound
from yogi import Yogi
from adamod import AdaMod
from Adan import Adan
from AdaGC import AdaGC
n_epochs = 30
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 50
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('F/mnist', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('F/mnist', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


network = Net()

optimizer = AdaGC(network.parameters())
#optimizer = optim.Adam(network.parameters(),lr = 0.001)
print(optimizer)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
time_train = []
time_test = []
target_accuracy=98.5   
start =time.time()
acc=[]
def train(epoch):
    network.train()
    train_time_1 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
    train_time_2 = time.time()
    print('该轮训练时间：{}s'.format(train_time_2-train_time_1))
    time_train.append(train_time_2-train_time_1)

def test():
    network.eval()
    test_time_1 = time.time()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_time_2 = time.time()
    time_test.append(test_time_2-test_time_1)
    print('该轮测试时间：{}s'.format(test_time_2-test_time_1))

    acc.append(100.* correct/len(test_loader.dataset))
    print('           第 {} 轮'.format(epoch))
    print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))
    print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
    if  acc[epoch-1] >= target_accuracy:
        use_time = time.time()
        time_taken = use_time - start
        print("达到精度 {}% 所用的时间：{} 秒".format(target_accuracy, time_taken))
        break

end = time.time()
print('总时间：',end - start,'s')
print('最大精度： {:.2f}%'.format(max(acc).item()))
print(time_train)
print(time_test)
