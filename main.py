import torch
import argparse
import os, cPickle
from model import NIP
from data_loader import data_loader
from train import *
from plot import plot

def main():
    parser = argparse.ArgumentParser(description='PyTorch  Training')
    parser.add_argument('--news_dim', default=500, type=int, help='feature dimension')
    parser.add_argument('--min_count', default=5, type=int, help='minimal frequency to filter words')
    parser.add_argument('--model_file', default='bitcoin_w2v_model', type=str, help='word2vector model')
    parser.add_argument('--news_file', default='dataset/Reuters_Bitcoin_News.xlsx', type=str, help='news content file')
    parser.add_argument('--price_file', default='dataset/price.csv', type=str, help='price file')
    parser.add_argument('--content_file', default=None, type=str, help='content to learn word2vec model')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--window_len', default=1, type=int, help='window size')
    parser.add_argument('--kernel_num', default=32, type=int, help='number of each kernel')
    parser.add_argument('--kernel_size', default='3,4,5', type=str, help='kernel size')
    parser.add_argument('--cnn_out_size', default=300, type=int, help='cnn output size')
    parser.add_argument('--lstm_hidden_size', default=100, type=int, help='lstm hidden size')
    parser.add_argument('--gru_hidden_size', default=100, type=int, help='gru hidden size')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout probability')
    parser.add_argument('--ratio', default=0.8, type=float, help='ratio of training dataset')
    parser.add_argument('--mlp_hidden', default='64,32', type=str, help='mlp hidden size')
    parser.add_argument('--num_classes', default=3, type=int, help='the number of classes')
    parser.add_argument('--max_epochs', default=50, type=int, help='the number of epochs')
    parser.add_argument('--dataset_dir', default='dataset/', type=str, help='directory of dataset')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    args = parser.parse_args()

    args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
    args.mlp_hidden = [int(k) for k in args.mlp_hidden.split(',')]
    use_cuda = torch.cuda.is_available() and args.cuda_able
    # use_cuda = False
    print 'use_cuda = ', use_cuda

    # build network
    print 'build network'
    net = NIP(args)

    # with open('test1.pkl','r') as f:
    #     x1,x2,y = cPickle.load(f)
    # train(net, (x1,x2,y), use_cuda, args, None)

    # load data
    # give each news several targets: 1h, 3h, 6h, 12h, 24h, 48h, 3days, 5days, 1week
    time_gaps = ['1h', '3h', '6h', '12h', '1d', '2d', '3d', '5d', '1w']
    kwargs = {'dataset_file': 'dataset_'+str(args.ratio)+'pkl',
              'target_news_file': 'news_info.pkl',
              'all_price_fluc_file': 'all_flucts.pkl',
              'target_price_fluc_file': 'price_info.pkl'}
    dataset = 'dataset_bs%d_ws%d_file' % (args.batch_size, args.window_len)
    if os.path.exists(dataset):
        print 'Loading dataset from %s' % dataset
        with open(dataset, 'r') as f:
            train_dataset, test_dataset = cPickle.load(f)
    else:
        train_dataset, test_dataset = data_loader(args, time_gaps, **kwargs)
        with open(dataset, 'w') as f:
            cPickle.dump((train_dataset, test_dataset), f)

    # for each time_interval, train a model
    for tg in time_gaps:
        # build network
        net = NIP(args)
        print 'Target time_gap is %s' % tg
        # training and testing
        train_acc, test_acc = [], []
        for epoch in range(args.max_epochs):
            print 'Epoch %d' %epoch
            train_acc.append(train(net, train_dataset, use_cuda, args, tg))
            test_acc.append(test(net, test_dataset, use_cuda, args.window_len, tg))

        pic_name = 'pics/acc_%s_%f.png' % (tg, args.ratio)
        plot(train_acc, test_acc, tg, pic_name)


if __name__ == '__main__':
    main()