import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def train(net, train_data, use_cuda, args, time_gap):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    batch_size = args.batch_size
    inputs, prices, targets = train_data
    num_samples = len(inputs)
    if args.window_len == 1:
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3])
        prices = prices.reshape(prices.shape[0],prices.shape[2])
    if time_gap is not None:
        targets = np.array(targets[time_gap])

    train_loss = 0
    total, correct = 0, 0

    for batch_idx in range(num_samples-batch_size+1):
        x1 = inputs[batch_idx: batch_idx+batch_size]
        x2 = prices[batch_idx: batch_idx+batch_size]
        y = targets[batch_idx: batch_idx+batch_size]
        if use_cuda:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        x1, x2, y = Variable(torch.Tensor(x1)), Variable(torch.Tensor(x2)), Variable(torch.LongTensor(y))
        optimizer.zero_grad()
        outputs = net(x1, x2)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


        train_loss += loss[0]
        _, predicted = torch.max(outputs.data, 1)

        total += x1.size(0)
        correct += predicted.eq(y.data).cpu().sum()
    print 'train loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        train_loss / (num_samples - batch_size + 1), 100.0 * correct / total, correct, total)
    # print 'Train loss = ', train_loss
    return float(correct)/ total

def test(net, test_data, use_cuda, window_len, time_gap):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    # transform data to Variable
    inputs, prices, targets = test_data
    if window_len == 1:
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3])
        prices = prices.reshape(prices.shape[0],prices.shape[2])
    targets = np.array(targets[time_gap])
    if use_cuda:
        inputs, prices, targets = inputs.cuda(), prices.cuda(), targets.cuda()
    inputs, prices, targets = Variable(torch.Tensor(inputs)), Variable(torch.Tensor(prices)), Variable(torch.LongTensor(targets))
    # test data with net
    outputs = net(inputs, prices)
    loss = criterion(outputs, targets)

    _, predicted = torch.max(outputs.data, 1)
    total = len(inputs)
    correct = predicted.eq(targets.data).cpu().sum()

    print 'Test loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        loss.data[0], 100.0 * correct / total, correct, total)
    # print 'Test loss = ', loss.data[0]
    # return outputs
    return float(correct) / total























