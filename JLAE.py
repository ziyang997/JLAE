# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import time
import math
import evaluate
from keras.layers import Lambda, Input, Dense
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from preprocessor import *
from test import parse_args
import threading
import os
from tensorflow.python.client import device_lib
import numpy as np

args = parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
data_generator = data(args.batch_size)
intermediate_dim = 512
latent_dim = 160


class JoVA():
    def __init__(self, args, data):
        self.args = args
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.train_R = data.train_R
        self.coffi = data.coffi
        self.test_R = data.test_R
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch_u = int(math.ceil(self.num_users / float(self.batch_size)))
        self.lr = args.lr
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.drop = tf.placeholder(tf.float32)
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")
        self.train_coffi = tf.nn.embedding_lookup(self.coffi, self.users)
        self.loss = self.foward()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def foward(self):
        # encoder
        #self.W_u = tf.Variable(tf.truncated_normal(shape=[self.num_items, latent_dim], mean=0.0,stddev=0.01), dtype=tf.float32, name="W_u")
        self.W_u = tf.get_variable('W_u', [self.num_items, latent_dim], tf.float32, xavier_initializer(seed=2021))
        self.W_i = tf.get_variable('W_i', [latent_dim, self.num_items], tf.float32, xavier_initializer(seed=2021))
        #self.W_i = tf.Variable(tf.truncated_normal(shape=[latent_dim, self.num_items], mean=0.0,stddev=0.01), dtype=tf.float32, name="W_i")
        u_embedding = tf.matmul(tf.nn.dropout(tf.square(tf.nn.l2_normalize(self.input_R_U, 1)),self.drop), self.W_u)
        self.decoder = tf.matmul(u_embedding,self.W_i)
        loss = tf.reduce_sum(tf.square(self.decoder-self.input_R_U)*self.train_coffi) 
        return loss
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model = JoVA(args, data_generator)
    np.random.seed(2021)
    tf.set_random_seed(2021) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, args.train_epoch + 1):
        t1 = time.time()
        loss = 0.
        random_row_idx = np.random.permutation(model.num_users)

        for i in range(model.num_batch_u):
            if i == model.num_batch_u - 1:
                row_idx = random_row_idx[i * model.batch_size:]
            else:
                row_idx = random_row_idx[(i * model.batch_size):((i + 1) * model.batch_size)]
            input_R_U = model.train_R[row_idx, :]
            _, batch_loss = sess.run(
                    [model.optimizer, model.loss],
                    feed_dict={model.users: row_idx, model.input_R_U: input_R_U,model.drop:0.3})

            loss += batch_loss / model.num_users
        
        print("Epoch %d //" % (epoch), " cost = {:.8f}".format(loss), "Elapsed time : %d sec" % (time.time() - t1))
        evaluate.test_all(sess, model)
        print("=" * 100)
