import os
import numpy as np
import cPickle
import pandas as pd
from Preprocess import news_loader, price_loader, get_w2v_model

def data_division(data, ratio, window_len, time_gaps):

    print 'start data division.'
    inputs, prices, targets = data

    x1sample, x2sample = [], []
    num_samples = len(targets)-window_len+1
    for i in range(num_samples):
        x1sample.append(inputs[i: i+window_len])
        x2sample.append(prices[i: i+window_len])

    ysample = targets[:num_samples]

    # to divide train and test dataset
    num_train = int(num_samples*ratio)

    train_news_samples = np.array(x1sample[:num_train])
    train_price_samples = np.array(x2sample[:num_train])
    train_targets = ysample[:num_train]

    test_news_samples = np.array(x1sample[num_train:])
    test_price_samples = np.array(x2sample[num_train:])
    test_targets = ysample[num_train:]

    train_dataset = (train_news_samples, train_price_samples, train_targets)
    test_dataset = (test_news_samples, test_price_samples, test_targets)
    return train_dataset, test_dataset


def data_loader(args, time_gaps, **kwargs):
    if kwargs is not None:
        keys = kwargs.keys()
        if 'dataset_file' in keys:
            dataset_file = args.dataset_dir + kwargs['dataset_file']
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r') as f:
                    train, test = cPickle.load(f)
                return train, test

        if 'target_news_file' in keys:
            target_news_file = args.dataset_dir + kwargs['target_news_file']
        else:
            target_news_file = None
        if 'all_price_fluc_file' in keys:
            all_price_fluc_file = args.dataset_dir + kwargs['all_price_fluc_file']
        else:
            all_price_fluc_file = None
        if 'target_price_fluc_file' in keys:
            target_price_fluc_file = args.dataset_dir + kwargs['target_price_fluc_file']
        else:
            target_price_fluc_file = None


    w2v_model = get_w2v_model(args.model_file, args.news_dim, args.min_count, args.content_file)
    news_matrix, time_list = news_loader(args.news_file, w2v_model, target_news_file)
    price_features, labels = price_loader(args.price_file, time_list, time_gaps, all_price_fluc_file, target_price_fluc_file)

    train, test = data_division((news_matrix, price_features, labels), args.ratio, args.window_len, time_gaps)
    return train, test