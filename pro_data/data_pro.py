'''
Data pre process part2
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
'''

import numpy as np
import re
import itertools
from collections import Counter

import tensorflow as tf
import csv
import os
import pickle
tf.flags.DEFINE_string("valid_data","../data/music/music_valid.csv", " Data for validation")
tf.flags.DEFINE_string("test_data", "../data/music/music_test.csv", "Data for testing")
tf.flags.DEFINE_string("train_data", "../data/music/music_train.csv", "Data for training")
tf.flags.DEFINE_string("user_review", "../data/music/user_review", "User's reviews")
tf.flags.DEFINE_string("item_review", "../data/music/item_review", "Item's reviews")
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(u_text,u_len,padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length=u_len
    u_text2={}
    print len(u_text)
    for i in u_text.keys():
        #print i
        sentence = u_text[i]
        if sequence_length>len(sentence):

            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            u_text2[i]=new_sentence
        else:
            new_sentence = sentence[:sequence_length]
            u_text2[i] = new_sentence

    return u_text2
def build_vocab(sentences1,sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1,vocabulary2, vocabulary_inv2]

def build_input_data(u_text,i_text, vocabulary_u,vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([vocabulary_u[word] for word  in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([vocabulary_i[word] for word in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def load_data(train_data,valid_data,user_review,item_review):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text,i_text, y_train, y_valid,u_len,i_len,uid_train,iid_train,uid_valid,iid_valid,user_num,item_num=\
        load_data_and_labels(train_data,valid_data,user_review,item_review)
    print "load data done"
    u_text = pad_sentences(u_text,u_len)
    print "pad user done"
    i_text=pad_sentences(i_text,i_len)
    print "pad item done"

    user_voc = [x for x in u_text.itervalues()]
    item_voc = [x for x in i_text.itervalues()]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print len(vocabulary_user)
    print len(vocabulary_item)
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)

    return [u_text,i_text,y_train,y_valid, vocabulary_user, vocabulary_inv_user,vocabulary_item,vocabulary_inv_item,
            uid_train, iid_train, uid_valid, iid_valid, user_num, item_num]


def load_data_and_labels(train_data,valid_data,user_review,item_review):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    f_train = open(train_data, "r")
    f1=open(user_review)
    f2=open(item_review)

    user_reviews=pickle.load(f1)
    item_reviews=pickle.load(f2)
    uid_train = []
    iid_train = []
    y_train=[]
    u_text = {}
    i_text = {}
    i = 0
    for line in f_train:
        i = i + 1
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            a=1
        else:
            u_text[int(line[0])] = '<PAD/>'
            for s in user_reviews[int(line[0])]:
                u_text[int(line[0])] = u_text[int(line[0])] + ' ' + s.strip()
            u_text[int(line[0])]=clean_str(u_text[int(line[0])])
            u_text[int(line[0])]=u_text[int(line[0])].split(" ")

        if i_text.has_key(int(line[1])):
            a=1
        else:
            i_text[int(line[1])] = '<PAD/>'
            for s in item_reviews[int(line[1])]:
                i_text[int(line[1])] = i_text[int(line[1])] + ' ' + s.strip()
            i_text[int(line[1])]=clean_str(i_text[int(line[1])])
            i_text[int(line[1])]=i_text[int(line[1])].split(" ")
        y_train.append(float(line[2]))
    print "valid"

    uid_valid = []
    iid_valid = []
    y_valid=[]
    f_valid=open(valid_data)
    for line in f_valid:
        line=line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if u_text.has_key(int(line[0])):
            a=1
        else:
            u_text[int(line[0])] = '<PAD/>'
            u_text[int(line[0])]=clean_str(u_text[int(line[0])])
            u_text[int(line[0])]=u_text[int(line[0])].split(" ")


        if i_text.has_key(int(line[1])):
            a=1
        else:
            i_text[int(line[1])] = '<PAD/>'
            i_text[int(line[1])]=clean_str(i_text[int(line[1])])
            i_text[int(line[1])]=i_text[int(line[1])].split(" ")

        y_valid.append(float(line[2]))
    print "len"
    u = np.array([len(x) for x in u_text.itervalues()])
    x = np.sort(u)
    u_len = x[int(0.85* len(u)) - 1]


    i = np.array([len(x) for x in i_text.itervalues()])
    y = np.sort(i)
    i_len = y[int(0.85 * len(i)) - 1]
    print "u_len:",u_len
    print "i_len:",i_len
    user_num = len(u_text)
    item_num = len(i_text)
    print "user_num:", user_num
    print "item_num:", item_num
    return [u_text,i_text,y_train,y_valid,u_len,i_len,uid_train,iid_train,uid_valid,iid_valid,user_num,item_num]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    TPS_DIR = '../data/music'
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    u_text,i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
    vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review)

    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train=list(zip( userid_train, itemid_train, y_train))
    batches_test=list(zip(userid_valid,itemid_valid,y_valid))
    output = open(os.path.join(TPS_DIR, 'music.train'), 'wb')
    pickle.dump(batches_train,output)
    output = open(os.path.join(TPS_DIR, 'music.test'), 'wb')
    pickle.dump(batches_test,output)

    para={}
    para['user_num']=user_num
    para['item_num']=item_num
    para['user_length']=u_text[0].shape[0]
    para['item_length'] = i_text[0].shape[0]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length']=len(y_train)
    para['test_length']=len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    output = open(os.path.join(TPS_DIR, 'music.para'), 'wb')

    pickle.dump(para, output)










