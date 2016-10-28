#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This is an example of using recurrent neural networks over characters
for DBpedia dataset to predict class from description of an entity.
This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626
and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import learn, metrics, framework

class RNNModel:
    def __init__(self, feature_dims, hidden_units = None, layers=1, model_dir = None):
        self.model_dir = model_dir

        x = tf.placeholder(tf.float32, [None, None, feature_dims])  
        y = tf.placeholder(tf.float32, [None, 1])
        length = tf.placeholder(tf.int32, [None]) 
        self.x, self.y, self.length = x, y, length

        cell = tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True)
        if layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers, state_is_tuple=True)

        output, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=x,
            dtype=tf.float32,
            sequence_length=length,
        )

        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)


        logit = tf.contrib.layers.fully_connected(inputs=relevant, 
            num_outputs=1, 
            activation_fn=None,
            biases_initializer=None
        )

        predictions = tf.sigmoid(logit) 

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logit, y)

        auc, update_auc = metrics.streaming_auc(predictions, y, num_thresholds = 10)
        stream_loss, update_stream_loss = metrics.streaming_mean(loss) 

        self.update_metrics = [update_stream_loss, update_auc]
        self.summaries = [auc, stream_loss]
        self.summary_labels = ['auc', 'loss']
        self.predictions = predictions

        self.learning_rate = tf.placeholder(tf.float32, [])
        self._global_step = framework.create_global_step()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=self._global_step)
        self.saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.initialize_local_variables())
        sess.run(tf.initialize_all_variables())
        self.sess = sess

    @property
    def global_step(self):
        return self.sess.run(self._global_step)

    def init_streaming(self):
        self.sess.run(tf.initialize_local_variables())

    def get_summaries(self):
        result = self.sess.run(self.summaries, feed_dict=None)
        return dict(zip(self.summary_labels, result))

    def val(self, x, y, length):
        self.sess.run(self.update_metrics, feed_dict={
            self.x : x,
            self.y : y,
            self.length : length,
        })

    def train(self, x, y, length, learning_rate):
        result = self.sess.run(self.summaries + self.update_metrics + [self.train_op], feed_dict={
            self.x : x,
            self.y : y,
            self.length : length,
            self.learning_rate : learning_rate
        })
        return dict(zip(self.summary_labels, result))

    def predict(self, x, length):
        return self.sess.run(self.predictions, feed_dict={
            self.x : x,
            self.length : length,
        })

    def save_checkpoint(self):
        assert self.model_dir is not None
        self.saver.save(self.sess, os.path.join(self.model_dir, 
            'model{}.ckpt'.format(self.global_step)))

    def restore(self, checkpoint):
        self.saver.restore(self.sess, checkpoint)





