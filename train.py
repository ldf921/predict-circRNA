from reader import process_feature
import random 
import numpy as np
import tensorflow as tf
from model import RNNModel
from sklearn import model_selection 
import logging

def get_feature_label(data):
    length = np.array(list(map(lambda x : x.length, data)), dtype = np.int32)
    max_length = np.max(length) 
    feature_dim = data[0].feature_dim
    batch_size = len(data)
    x = np.zeros([batch_size, max_length, feature_dim])
    y = np.zeros([batch_size, 1])

    for i in range(len(data)):
        x[i, : data[i].length, :] = data[i].features
        y[i, 0] = data[i].label

    return x, y, length

def val(model, data, batch_size = 64):
    model.init_streaming()

    for i in range(0, len(data), batch_size):
        model.val(*get_feature_label(data[i:i+batch_size]))

    return model.get_summaries()    

    
def train(steps = 1000, val_per_steps = 100, batch_size = 20, learning_rate = 0.01):
    model = RNNModel(4)

    data = process_feature()
    kfolds = model_selection.KFold(n_splits = 10, shuffle = True, random_state = 233)
    train_index, val_index = next(iter(kfolds.split(data)))
    train_data, val_data = [data[i] for i in train_index], [data[i] for i in val_index]

    for t in range(0, steps):
        x, y, length = get_feature_label(random.sample(train_data, batch_size))
        result = model.train(x, y, length, learning_rate)
        logging.info("step = {}: {}".format(t, result))

        if (t + 1) % val_per_steps == 0:
            result = val(model, val_data)
            logging.info("validation for step = {}: {}".format(t, result))


logging.basicConfig(filename='train.log', level=logging.INFO)
train()