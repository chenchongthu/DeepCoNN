'''
DeepCoNN
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''

import numpy as np
import tensorflow as tf
import pickle
import os
import math
from tensorflow.contrib import learn
import datetime
import csv
import pickle

tf.flags.DEFINE_string("word2vec", "../data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data","../data/yelp/yelp.valid", " Data for validation")
tf.flags.DEFINE_string("test_data", "../data/yelp/yelp.test", "Data for testing")
tf.flags.DEFINE_string("train_data", "../data/yelp/yelp.train", "Data for training")
tf.flags.DEFINE_string("user_review", "../data/yelp/user_review", "User's reviews")
tf.flags.DEFINE_string("item_review", "../data/yelp/item_review", "Item's reviews")

# ==================================================

# Model Hyperparameters
#tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda")
tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularizaion V")
# Training parameters
tf.flags.DEFINE_integer("batch_size",100, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

class DeepCoNN(object):
    def __init__(
            self, user_length,item_length, num_classes, user_vocab_size,item_vocab_size,fm_k,n_latent,user_num,item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,l2_reg_V=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with  tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, user_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(1,pooled_outputs_u)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, item_length- filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(1,pooled_outputs_i)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i= tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, n_latent], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, n_latent], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf,self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf,self.input_iid)
            self.uid = tf.reshape(self.uid,[-1,n_latent])
            self.iid = tf.reshape(self.iid,[-1,n_latent])

            #self.uid=tf.nn.dropout(self.uid,self.dropout0)
            #self.iid=tf.nn.dropout(self.iid,self.dropout0)

            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas = tf.matmul(self.h_drop_u, Wu)+self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas = tf.matmul(self.h_drop_i, Wi) +self.iid+ bi

            #self.u_feas=tf.nn.relu(self.u_feas)
            #self.i_feas=tf.nn.relu(self.i_feas)
            #self.u_feas=tf.nn.dropout(self.u_feas,self.dropout_keep_prob)
            #self.i_feas=tf.nn.dropout(self.i_feas,self.dropout_keep_prob)
        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)
            
            self.FM=tf.nn.dropout(self.FM,self.dropout_keep_prob)

            Wmul=tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul=tf.matmul(self.FM,Wmul)
            self.score=tf.reduce_sum(self.mul,1,keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.01, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.01, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            l2_loss2=0
            l2_loss2 += tf.nn.l2_loss(self.uidW2)
            l2_loss2 +=tf.nn.l2_loss(self.iidW2)
            self.predictions = self.score + self.Feature_bias + self.bised
        with tf.name_scope("loss"):
            #losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))



if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")


    pkl_file = open('yelp.para', 'rb')

    para = pickle.load(pkl_file)
    user_num= para['user_num']
    item_num=para['item_num']
    user_length=para['user_length']
    item_length=para['item_length']
    vocabulary_user=para['user_vocab']
    vocabulary_item=para['item_vocab']
    train_length=para['train_length']
    test_length=para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    np.random.seed(2017)
    random_seed = 2017




    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep=DeepCoNN(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                num_classes=1,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                fm_k=8,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=16)
            tf.set_random_seed(random_seed)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=1e-8).minimize(deep.loss)

            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            '''optimizer=tf.train.RMSPropOptimizer(0.002)
            grads_and_vars = optimizer.compute_gradients(deep.loss)'''
            train_op = optimizer#.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            if FLAGS.word2vec:
                # initial matrix with random uniform
                u=0
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx=0

                        if word in vocabulary_user:
                            u=u+1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(deep.W1.assign(initW))
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))

                item=0
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx =0
                        if word in vocabulary_item:
                            item=item+1
                            idx = vocabulary_item[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                sess.run(deep.W2.assign(initW))

            def train_step(u_batch,i_batch,uid,iid, y_batch,batch_num):
                """
                A single training step
                """
                feed_dict = {
                    deep.input_u: u_batch,
                    deep.input_i: i_batch,
                    deep.input_y:y_batch,
                    deep.input_uid:uid,
                    deep.input_iid:iid,
                    deep.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy, mae = sess.run(
                    [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                
                #print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
                return accuracy,mae

            def dev_step(u_batch, i_batch,uid,iid,y_batch, writer=None):
                """
                Evaluates model on a dev set

                """
                feed_dict = {
                    deep.input_u: u_batch,
                    deep.input_i:i_batch,
                    deep.input_y: y_batch,
                    deep.input_uid:uid,
                    deep.input_iid:iid,
                    deep.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, mae = sess.run(
                    [global_step, deep.loss, deep.accuracy, deep.mae],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

                return [loss,accuracy,mae]

            l = (train_length / FLAGS.batch_size) + 1
            print l
            ll = 0
            epoch = 1
            best_mae=5
            best_rmse=5
            train_mae = 0
            train_rmse = 0

            pkl_file = open('yelp.train', 'rb')

            train_data = pickle.load(pkl_file)

            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open('yelp.test', 'rb')

            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_test = len(test_data)
            batch_size = 100
            ll = int(len(train_data) / batch_size)


            for epoch in range(40):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train=shuffled_data[start_index:end_index]

                    uid, iid, y_batch=zip(*data_train)

                    u_batch = []
                    i_batch = []
                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)

                    t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch,batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae


                    if batch_num % 1000 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print batch_num
                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0

                        ll_test = int(len(test_data) / batch_size) + 1
                        for batch_num2 in range(ll_test):
                            start_index = batch_num2 * batch_size
                            end_index = min((batch_num2 + 1) * batch_size, data_size_test)
                            data_test = test_data[start_index:end_index]

                            userid_valid, itemid_valid, y_valid = zip(*data_test)

                            u_valid = []
                            i_valid = []
                            for i in range(len(userid_valid)):
                                u_valid.append(u_text[userid_valid[i][0]])
                                i_valid.append(i_text[itemid_valid[i][0]])
                            u_valid = np.array(u_valid)
                            i_valid = np.array(i_valid)


                            loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                            loss_s = loss_s + len(u_valid) * loss
                            accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                            mae_s = mae_s + len(u_valid) * mae
                        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                         np.sqrt(accuracy_s / test_length),
                                                                                         mae_s / test_length))


                print str(epoch) + ':\n'
                print("\nEvaluation:")
                print "train:rmse,mae:", train_rmse / ll, train_mae / ll
                train_rmse = 0
                train_mae = 0

                loss_s = 0
                accuracy_s = 0
                mae_s = 0

                ll_test=int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test=test_data[start_index:end_index]

                    userid_valid, itemid_valid, y_valid=zip(*data_test)
                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuracy, mae=dev_step(u_valid,i_valid,userid_valid,itemid_valid, y_valid)
                    loss_s = loss_s + len(u_valid) * loss
                    accuracy_s = accuracy_s + len(u_valid) * np.square(accuracy)
                    mae_s = mae_s + len(u_valid) * mae
                print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                             np.sqrt(accuracy_s /test_length),
                                                                                             mae_s / test_length))
                rmse=np.sqrt(accuracy_s /test_length)
                mae=mae_s / test_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print 'best rmse:',best_rmse
            print 'best mae:',best_mae



    print 'begin'
