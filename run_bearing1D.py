import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import hickle as hkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import urllib
import os
import tarfile
import skimage
from tensorflow.keras import layers
import skimage.io
import skimage.transform
from flip_gradient import flip_gradient
from utils import *
from modelsbearing import *
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
#Loading Mnist Data
print ("LOADING PU-0")
ratio = [0.9,0.1]
def random_shuffle(data,label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data,label

enc = OneHotEncoder()
fl = loadmat('data/CWRUSTFT0.mat')
data_s = fl['data'].reshape(-1,33,33,1).astype(np.float32)
label_s = fl['label'].reshape(-1,1)
#打乱顺序
data_s,label_s = random_shuffle(data_s,label_s)
#one-hot
enc.fit(label_s)
label_s = enc.transform(label_s).toarray()
#划分数据集
tmp = int(len(label_s)*ratio[0])
data_s_train = data_s[:tmp]
label_s_train = label_s[:tmp]
data_s_test = data_s[tmp:]
label_s_test = label_s[tmp:]

#Loading MnistM Data
print ("LOADING PU-1")
fl = loadmat('data/CWRUSTFT0.mat')
data_t = fl['data'].reshape(-1,33,33,1).astype(np.float32)
label_t = fl['label'].reshape(-1,1)
#打乱顺序
data_t,label_t = random_shuffle(data_t,label_t)
#one-hot
enc.fit(label_t)
label_t = enc.transform(label_t).toarray()
#划分数据集
tmp = int(len(label_t)*ratio[0])
data_t_train = data_t[:tmp]
label_t_train = label_t[:tmp]
data_t_test = data_t[tmp:]
label_t_test = label_t[tmp:]

print ("GENERATING DOMAIN DATA")

# PLOT IMAGES
imshow_grid(data_s, shape=[5, 10],title='Source PU-0')
imshow_grid(data_t, shape=[5, 10],title='Target PU-1')

# Create a mixed dataset for TSNE visualization
num_test = 300
combined_test_imgs = np.vstack([data_s_test[:num_test], data_t_test[:num_test]])
combined_test_labels = np.vstack([label_s_test[:num_test], label_t_test[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),np.tile([0., 1.], [num_test, 1])])


class DSNModel(object):
    """ Domain separation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        # define the place holders
        self.source = tf.placeholder(tf.uint8, [None, 33, 33, 1], name='input_s')
        self.target = tf.placeholder(tf.uint8, [None, 33, 33, 1], name='input_t')

        self.y_s = tf.placeholder(tf.float32, [None, 10], name='labels_s')
        self.y_t = tf.placeholder(tf.float32, [None, 10], name='labels_t')

        self.d = tf.placeholder(tf.float32, [None, 2])  # DOMAIN LABEL
        self.dw = tf.placeholder(tf.float32, [])

        self.X_input_s = tf.cast(self.source, tf.float32)
        self.X_input_t = tf.cast(self.target, tf.float32)

        # source shared encoder
        with tf.variable_scope('shared_encoder')as shared:
            self.shared_s = shared_encoder(self.X_input_s, name='shared_encoder')

        # target common encoder
        with tf.variable_scope(shared, reuse=True):
            self.shared_t = shared_encoder(self.X_input_t, name='shared_encoder', reuse=True)

        # target private encoder
        with tf.variable_scope('priviate_target_encoder'):
            self.private_t = private_target_encoder(self.X_input_t, name='target_private')

        # source private encoder
        with tf.variable_scope('priviate_source_encoder'):
            self.private_s = private_source_encoder(self.X_input_s, name='source_private')

        # lables predictor
        with tf.variable_scope('label_predictor') as pred:
            logits_s = slim.fully_connected(self.shared_s, 100, scope='fc1')
            logits_s = slim.fully_connected(logits_s, 64, scope='fc2')
            logits_s = slim.fully_connected(logits_s, 32, scope='fc3')
            logits_s = slim.fully_connected(logits_s, 16, scope='fc4')
            self.logits_s = slim.fully_connected(logits_s, 10, activation_fn=None, scope='out')

        with tf.variable_scope(pred, reuse=True):
            logits_t = slim.fully_connected(self.shared_t, 100, scope='fc1')
            logits_t = slim.fully_connected(logits_t, 64, scope='fc2')
            logits_t = slim.fully_connected(logits_t, 32, scope='fc3')
            logits_t = slim.fully_connected(logits_t, 16, scope='fc4')
            self.logits_t = slim.fully_connected(logits_t, 10, activation_fn=None, scope='out')

        # domain predictor

        self.shared_feat = layers.concatenate([self.shared_s, self.shared_t], name='merged_features', axis=0)
        with tf.variable_scope('domain_predictor'):
            logits_d = flip_gradient(self.shared_feat, 0.5)  # GRADIENT REVERSAL
            logits_d = slim.fully_connected(logits_d, 100, scope='d1')
            self.logits_d = slim.fully_connected(logits_d, 2, activation_fn=None, scope='out')

        # Shared decoder
        # input for decoder
        self.target_concat_feat = concat_operation(self.shared_t, self.private_t)
        self.source_concat_feat = concat_operation(self.shared_s, self.private_s)

        with tf.variable_scope('shared_decoder') as decoder:
            self.decode_s = shared_decoder(self.source_concat_feat, 33, 33, 1)

        with tf.variable_scope(decoder, reuse=True):
            self.decode_t = shared_decoder(self.target_concat_feat, 33, 33, 1)


# Build the model graph
tf.reset_default_graph()
graph = tf.get_default_graph()
with graph.as_default():
    model = DSNModel()
    # lr = tf.placeholder(tf.float32, []) #learning rate

    # classification loss
    source_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.logits_s, labels=model.y_s))

    # similarity loss
    similarity_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.logits_d, labels=model.d))

    # difference loss
    target_diff_loss = difference_loss(model.shared_t, model.private_t, 0.05)
    source_diff_loss = difference_loss(model.shared_s, model.private_s, 0.05)
    diff_loss = target_diff_loss + source_diff_loss
    # reconstruction loss
    target_recon_loss = tf.contrib.losses.mean_pairwise_squared_error(model.X_input_t,
                                                                      model.decode_t) * 1e-3  # tf.contrib.losses.mean_pairwise_squared_error(target,target_recon,1e-6)
    source_recon_loss = tf.contrib.losses.mean_pairwise_squared_error(model.X_input_s,
                                                                      model.decode_s) * 1e-3  # tf.contrib.losses.mean_pairwise_squared_error(source,source_recon,1e-6)#
    recon_loss = target_recon_loss + source_recon_loss

    # total loss
    total_loss = source_class_loss + similarity_loss + recon_loss + diff_loss
    # optimization
    optm_class_dann = tf.train.AdamOptimizer(0.005).minimize(total_loss)

    # Evaluation

    # source accuracy
    source_pred = tf.nn.softmax(model.logits_s)
    correct_source_pred = tf.equal(tf.argmax(source_pred, 1), tf.argmax(model.y_s, 1))
    source_acc = tf.reduce_mean(tf.cast(correct_source_pred, tf.float32))

    # target accuracy
    target_pred = tf.nn.softmax(model.logits_t)
    correct_target_pred = tf.equal(tf.argmax(target_pred, 1), tf.argmax(model.y_t, 1))
    target_acc = tf.reduce_mean(tf.cast(correct_target_pred, tf.float32))

    # domain accuracy
    domain_pred = tf.nn.softmax(model.logits_d)
    correct_domain_pred = tf.equal(tf.argmax(domain_pred, 1), tf.argmax(model.d, 1))
    accr_domain_dann = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
    print("MODEL BUILT SUCCESSFULLY!!!")


def train_and_evaluate(training_mode, graph, model, batch_size=128, every_epoch=1, num_epochs=300, verbose=True):
    """Helper to run the model with different training modes."""

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Batch generators
    # Batch generators
    gen_source_batch = batch_generator(
        [data_s_train, label_s_train], batch_size // 2)
    gen_target_batch = batch_generator(
        [data_t_train, label_t_train], batch_size // 2)

    domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                               np.tile([0., 1.], [batch_size // 2, 1])])
    ntrain = data_s_train.shape[0]
    num_batch = int(ntrain / batch_size)
    # Training loop
    num_steps = num_epochs * num_batch
    for i in range(num_steps):

        p = float(i) / num_steps
        l = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.0001 / (1. + 10 * p) ** 0.75

        # Training step
        if training_mode == 'DSN':

            batch_source, batch_y_s = next(gen_source_batch)
            batch_target, batch_y_t = next(gen_target_batch)

            feeds_class = {model.source: batch_source, model.target: batch_target,
                           model.y_s: batch_y_s, model.d: domain_labels, model.dw: l}

            _, domain_acc, recon_l, source_loss, diff = sess.run(
                [optm_class_dann, accr_domain_dann, recon_loss, source_class_loss, diff_loss], feed_dict=feeds_class)
            if verbose and i % 5 == 0:
                # CHECK BOTH LOSSES
                print("[%d/%d] p: %.3f d_acc: %.3f, class_loss: %.3f, recon_loss: %.3f, diff_loss: %.3f"
                      % (i, num_steps, p, domain_acc, source_loss, recon_l, diff))
                # CHECK ACCUARACIES OF BOTH SOURCE AND TARGET
                feed_source = {model.source: data_s_test, model.y_s: label_s_test}
                feed_target = {model.target: data_t_test, model.y_t: label_t_test}
                accr_source_dann = sess.run(source_acc, feed_dict=feed_source)
                accr_target_dann = sess.run(target_acc, feed_dict=feed_target)
                print(" DANN: SOURCE ACCURACY: %.3f TARGET ACCURACY: %.3f" % (accr_source_dann, accr_target_dann))

    # shared features embeddings
    test_emb = sess.run(model.shared_feat,
                        feed_dict={model.source: data_s_test[:num_test], model.target: data_t_test[:num_test]})

    return test_emb, sess

print('\n Adaptation training')
dsn_emb,sess = train_and_evaluate('DSN', graph, model)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dsn_tsne = tsne.fit_transform(dsn_emb)
plot_embedding(dsn_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Domain Adaptation')

def compare_recon(sess, batch_source, batch_target, model):
    s = sess.run([model.decode_s], feed_dict={model.source: batch_source, model.target: batch_target})
    reconstructed = [(i - i.min()) / (i.max() - i.min()) for i in s[0]]
    # plt.title('Training Source Images')
    imshow_grid(batch_source, shape=[8, 8], title='Training Source Images')
    # plt.title('Reconstructed Source Images')
    imshow_grid(reconstructed, shape=[8, 8], title='Reconstructed Source Images')

    s = sess.run([model.decode_t], feed_dict={model.source: batch_source, model.target: batch_target})
    # reconstructed=(s[0]-s[0].min())/(s[0].max()-s[0].min())
    reconstructed = [(i - i.min()) / (i.max() - i.min()) for i in s[0]]
    # plt.title('Reconstructed Target Images')
    imshow_grid(batch_target, shape=[8, 8], title='Training Target Images')

    imshow_grid(reconstructed, shape=[8, 8], title='Reconstructed Target Images')
    # plt.title('Training Target Images')

compare_recon(sess,data_s_test[:64],data_t_test[:64],model)