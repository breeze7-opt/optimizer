import argparse
import time
import math
import torch
import torch.nn as nn
import corpus
import RNN
from AdaBound import AdaBound
from yogi import Yogi
from adamod import AdaMod
from Adan import Adan
import time
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='F/input', # /input
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--save', type=str,  default='F/output/model_test.pt')

args = parser.parse_args()

device = torch.device("cuda")
corpus = corpus.Corpus(args.data)
# Build the model
interval = 200 # interval to report
ntokens = len(corpus.dictionary) # 10000
model = RNN.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)

model = model.to(device)

from AdaGC import AdaGC
optimizer = AdaGC(model.parameters())
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load data

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size) # size(total_len//bsz, bsz)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

#以上均为预处理

# Load checkpoint
if args.checkpoint != '':
    model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

print(model)
criterion = nn.CrossEntropyLoss()

criterion = criterion.to(device)

# Training code

def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    # source: size(total_len//bsz, bsz)
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        model.eval()
        test_time_1 = time.time()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            data = data.to(device)
            targets = targets.to(device)
            output, hidden = model(data, hidden)
           
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        test_time_2 = time.time()
        time_test.append(test_time_2-test_time_1)
        print('该轮测试时间：{}s'.format(test_time_2-test_time_1))
        return total_loss / len(data_source)


def train():

    model.train()
    train_time_1 = time.time()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        data = data.to(device)
        targets = targets.to(device)
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch % interval == 0 and batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            total_loss = 0
            start_time = time.time()
    train_time_2 = time.time()
    print('该轮训练时间：{}s'.format(train_time_2-train_time_1))
    time_train.append(train_time_2-train_time_1)
# Loop over epochs.
lr = args.lr
best_val_loss = None
start = time.time()
test_ppl = []
time_train = []
time_test = []
target_accuracy=104  
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        test_loss = evaluate(test_data)
        print('-' * 30)
        test_ppl.append(math.exp(test_loss))
        if  test_ppl[epoch-1] <= target_accuracy:
            use_time = time.time()
            time_taken = use_time - start
            print("达到精度 {} 所用的时间：{} 秒".format(target_accuracy, time_taken))
            break
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           test_loss, math.exp(test_loss)))
        print('-' * 30)
    print('最小ppl为:{:8.2f}'.format(min(test_ppl)))
    print(time_train)
    print(time_test)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

