from reader import process_feature, read_data
import random 
import numpy as np
import tensorflow as tf
from model import RNNModel
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score
import os
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from operator import attrgetter
import math

def get_feature_label(data, length_limit = math.inf):
    data = list(filter(lambda d: d.seq is not None, data))
    for i in range(len(data)):
        if not hasattr(data[i], 'features'):
            data[i].get_feature()    
    

    length = np.array(list(map(lambda x : x.length, data)), dtype = np.int32)
    # print(length)
    max_length = min(np.max(length), length_limit)
    feature_dim = data[0].feature_dim
    batch_size = len(data)
    x = np.zeros([batch_size, max_length, feature_dim])
    y = np.zeros([batch_size, 1])

    for i in range(len(data)):
        length[i] = min(length[i], length_limit)
        s = np.random.randint(data[i].length - length[i] + 1)
        x[i, : length[i], :] = data[i].features[s : s + length[i], :]
        y[i, 0] = data[i].label

    return x, y, length

def val(model, data, batch_size = 64):
    model.init_streaming()

    for i in range(0, len(data), batch_size):
        model.val(*get_feature_label(data[i:i+batch_size], length_limit = 1000))

    return model.get_summaries()

# class SimpleLengthModel:
#     threshold_length = 1000
#     @classmethod
#     def data_filter(cls, data):
#         return data.length <= cls.threshold_length 

def batch_data_provider(data, batch_size):    
    data_label = [] 
    for l in (0, 1):
        data_label.append(list(filter(lambda d : d.label == l, data)))
    samples_label = [int(batch_size / 2), int(batch_size / 2) ]

    while True:
        yield random.sample(data_label[0], samples_label[0]) + random.sample(data_label[1], samples_label[1])

def train(train_data, val_data, steps = 6000, val_per_steps = 300, checkpoint_per_steps=100, batch_size = 64, learning_rate = 0.01, **kwargs):
    global args

    # train_data = list(filter(SimpleLengthModel.data_filter, train_data))
    # val_data = list(filter(SimpleLengthModel.data_filter, val_data))

    model = RNNModel(feature_dims=train_data[0].feature_dim, model_dir=args.output_dir, **kwargs)
    if args.checkpoint is not None:
        model.restore(args.checkpoint)
    data_provider = batch_data_provider(train_data, batch_size=batch_size)

    for t in range(0, steps):
        x, y, length = get_feature_label(next(data_provider), length_limit=1000)
        result = model.train(x, y, length, learning_rate)
        logging.info("step = {}: {}".format(model.global_step, result))

        if model.global_step % val_per_steps == 0:
            result = val(model, val_data)
            model.init_streaming()
            logging.info("validation for step = {}: {}".format(model.global_step, result))
        if model.global_step % checkpoint_per_steps == 0:
            model.save_checkpoint()
            logging.info("save checkpoint at {}".format(model.global_step))

        if model.global_step % 2000 == 0:
            learning_rate *= 0.2 
            logging.info("current learning rate = {}".format(learning_rate))

def test(data, batch_size=64, filename='roc.png', **kwargs):
    global args

    assert args.checkpoint is not None
    model = RNNModel(feature_dims=data[0].feature_dim, model_dir=args.output_dir, **kwargs)
    model.restore(args.checkpoint)
    data = list(filter(lambda d: d.seq is not None, data))
    for i in tqdm(range(0, len(data), batch_size)):
        x, y, length = get_feature_label(data[i:i+batch_size], length_limit=10000)
        predictions = model.predict(x, length)
        for l,p in zip(data[i:i+batch_size], predictions):
            l.prediction = p 
        # if SimpleLengthModel.data_filter(data[i]):
            # x, y, length = get_feature_label(data[i:i+batch_size], length_limit=1000)
            # predictions = model.predict(x, length)
            # for l,p in zip(data[i:i+batch_size], predictions):
            #     l.prediction = p 
        # else:
        #     for l in data[i:i+batch_size]:
        #         l.prediction = 1 + l.length / 100000.0


    predictions = list(map(attrgetter('prediction'), data))
    labels = list(map(attrgetter('label'), data)) 

    plot_roc(predictions, labels, filename=filename)

def baseline(data, filename):
    labels = list(map(attrgetter('label'), data))
    predictions = list(map(attrgetter('length'), data))
    plot_roc(predictions, labels, filename=filename)

def plot_roc(predictions, labels, plot_samples = 50, filename='roc.png'):
    global args

    predictions = np.array(predictions)
    fpr, tpr, th = roc_curve(labels, predictions)

    plot_samples = min(plot_samples, len(fpr))
    indices = np.round(np.arange(0, plot_samples) * len(fpr) / plot_samples).astype(np.int32)

    th = th - np.min(th)
    th = th / np.max(th)
    
    r = auc(fpr, tpr)
    logging.info('Score AUC = {}'.format(r))
    f1_scores = [f1_score(labels, predictions > th) for th in tqdm(np.arange(0.1, 0.9, 0.01))]
    logging.info('Score F1 = {}'.format(np.max(f1_scores))) 
    logging.info('Score AP = {}'.format(average_precision_score(labels, predictions)))

    # plt.figure(figsize=(9, 7), dpi=96)
    # plt.plot(fpr[indices], tpr[indices], label="Model")
    # plt.plot([0, 1], [0, 1], linestyle='--', label="Guess")
    # plt.plot(fpr[indices], th[indices], label = "Threshold")
    # plt.legend()
    # plt.title('auc = {}'.format(r))
    # plt.savefig(os.path.join(args.output_dir, filename))
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('-o', '--output_dir', type=str, default='temp')
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('action', type=str, default=None)

    args = parser.parse_args()

    if args.checkpoint is not None:
        args.output_dir = os.path.dirname(args.checkpoint)
    try:
        os.mkdir(args.output_dir)
    except FileExistsError:
        assert os.path.isdir(args.output_dir), 'output_dir should be a directory'

    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), 
        format='[%(asctime)s] %(message)s',
        filemode='w' if args.checkpoint is None else 'a',
        level=logging.INFO)

    data = read_data()
    kfolds = model_selection.KFold(n_splits = 10, shuffle = True, random_state=args.seed)
    train_index, val_index = next(kfolds.split(data))
    train_data, val_data = [data[i] for i in train_index], [data[i] for i in val_index]
    train_data = list(filter(lambda l : l.length > 0, train_data))
    val_data = filter(lambda l : l.length > 0, val_data)
    val_data = list(sorted(val_data, key=attrgetter('length')))

    if args.action is None:
        if args.checkpoint is not None:
            args.action = 'test'
        else:
            args.action = 'train'

    model_dict = dict(hidden_units=50,layers=1)
    if args.model is not None:
        model_dict.update(eval('dict({})'.format(args.model)))
    if args.action == 'train':
        train(train_data, val_data, batch_size=50, learning_rate=0.01, **model_dict)
    elif args.action == 'baseline':
        # baseline(list(filter(SimpleLengthModel.data_filter, val_data)))
        lengths = np.array(list(map(attrgetter('length'), train_data)))
        print(np.min(lengths), np.max(lengths), np.mean(lengths))
        # baseline(val_data, filename='roc_baseline_full.png')
    elif args.action == 'test':
        # test(list(filter(SimpleLengthModel.data_filter, val_data)), filename='roc_1k.png')
        test(val_data, filename='roc_model.png', **model_dict)
        # val_data = list(filter(lambda x : x.lenRgth > 1000, val_data))
        # print(len(val_data))
        # test(val_data, filename='roc_gt1000.png')

