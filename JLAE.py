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
data_generator = ml1m(args.batch_size)
intermediate_dim = 512
latent_dim = 160


class JoVA():
    def __init__(self, args, data):
        self.args = args
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.max_item = data.max_item
        self.item_train = data.item_train
        self.train_R = data.train_R
        self.train_R_U = data.train_R_U
        self.train_R_I = data.train_R_I
        self.train_R_U_norm = tf.nn.l2_normalize(self.train_R_U, 1)
        self.train_R_I_norm = tf.nn.l2_normalize(self.train_R_I, 1)
        self.test_R = data.test_R
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size

        self.train_num = self.train_R.sum()
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.lr = args.lr

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")

        self.user_pos_item = tf.nn.embedding_lookup(self.item_train, self.users)

        self.u_embeddings = self.user_vae()
        self.i_embeddings = self.item_vae()

        #self.u_g_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)

        self.loss = self.get_loss()
        self.rating = self._pre()
        self.rating = tf.reshape(self.rating, [-1, self.num_items + 1])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def user_vae(self):
        # encoder

        self.W4_u = tf.get_variable('W4_u', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        self.h = tf.get_variable('h', [1, latent_dim], tf.float32, xavier_initializer())
        u_embedding_ae = tf.matmul(self.input_R_U, self.W4_u)
        u_embedding = u_embedding_ae

        return u_embedding

    def item_vae(self):
        # encoder

        self.W4_i = tf.get_variable('W4_i', [self.num_items + 1, latent_dim], tf.float32, xavier_initializer())

        i_embedding = self.W4_i

        return i_embedding

    def get_loss(self):
        pos_item = tf.nn.embedding_lookup(self.i_embeddings, self.user_pos_item)
        pos_num_r = tf.cast(tf.not_equal(self.user_pos_item, self.num_items), 'float32')
        pos_item = tf.einsum('ab,abc->abc', pos_num_r, pos_item)
        pos_r = tf.einsum('ac,abc->ab', self.u_embeddings, pos_item)
        pos_r = tf.reshape(pos_r, [-1, self.max_item])

        loss = 0.5 * tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', self.i_embeddings, self.i_embeddings), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', self.u_embeddings, self.u_embeddings), 0)
                          , 0), 0)
        loss += tf.reduce_sum((1.0 - 0.5) * tf.square(pos_r) - 2.0 * pos_r)

        return loss

    def _pre(self):
        pre = tf.matmul(self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True)
        return pre


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model = JoVA(args, data_generator)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # tf.set_random_seed(777)

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    for epoch in range(1, args.train_epoch + 1):
        t1 = time.time()
        loss = 0.
        random_row_idx = np.random.permutation(model.num_users)

        for i in range(model.num_batch):
            if i == model.num_batch - 1:
                row_idx = random_row_idx[i * model.batch_size:]
            else:
                row_idx = random_row_idx[(i * model.batch_size):((i + 1) * model.batch_size)]
                input_R_U = model.train_R[row_idx, :]
                _, batch_loss = sess.run(
                    [model.optimizer, model.loss],
                    feed_dict={model.users: row_idx, model.input_R_U: input_R_U})

            loss += batch_loss / model.num_batch

        print("Epoch %d //" % (epoch), " cost = {:.8f}".format(loss), "Elapsed time : %d sec" % (time.time() - t1))
        evaluate.test_all(sess, model)
        print("=" * 100)
