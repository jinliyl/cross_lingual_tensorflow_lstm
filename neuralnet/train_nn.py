import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from neuralnet.cross_cnn import cross_cnn
from neuralnet.single_cnn import single_cnn
from neuralnet.single_bi_lstm import single_bi_lstm

class train_nn():

    def __init__(self, emotion_list = [], target_dic_path = "", source_dic_path = "", target_path = "", source_path = "", 
                transfer_path = "", part = 1, model = "cnn", cn2en = 1, sequence_length = 150, cross_lingual = True,
                embedding_dim = 128, filter_sizes = [3, 4, 5], num_filters = 128, dropout_keep_prob = 0.5,
                l2_reg_lambda = 0.01, batch_size = 64, num_epochs = 200, evaluate_every = 20, checkpoint_every = 100):
        
        self.emotion_list = emotion_list
        self.target_dic_path = target_dic_path
        self.source_dic_path = source_dic_path
        self.target_path = target_path
        self.source_path = source_path
        self.transfer_path = transfer_path
        self.part = part

        self.model = model
        if self.model in ["cnn"]:
            self.dynamic = False
        elif self.model in ["lstm", "bi_lstm"]:
            self.dynamic = True
        else:
            self.dynamic = False

        self.cn2en = cn2en
        self.sequence_length = sequence_length
        self.cross_lingual = cross_lingual

        self.embedding_dim = embedding_dim
        self.filter_sizes =  filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
              
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every


    def run(self):
        print("emotions: " + str(self.emotion_list))
        if self.cross_lingual:
            self.load_cross_data()
            self.cross_training()
        else:
            self.load_single_data()
            self.single_training()

    def load_data(self, path, add_len = 0):
        data_label = []
        data_feature = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                label, line = line.split("\t")
                if label not in self.emotion_list:
                    continue
                labelfeature = [0 for ii in range(len(self.emotion_list))]
                pos = self.emotion_list.index(label)
                labelfeature[pos] = 1
                data_label.append(labelfeature)
                ll = line.split(" ")
                data_feature.append([(int(x) + add_len + 1) for x in ll])
                '''
                feature = []
                if len(ll) < self.sequence_length:
                    feature += [(int(x) + add_len + 1) for x in ll]
                else:
                    feature += [(int(x) + add_len + 1) for x in ll[:self.sequence_length]]
               
                if self.dynamic:
                    if len(ll) < self.sequence_length:
                        feature += [(int(x) + add_len + 1) for x in ll]
                    else:
                        feature += [(int(x) + add_len + 1) for x in ll[:self.sequence_length]]
                else:
                    if len(ll) < self.sequence_length: 
                        feature += [(int(x) + add_len + 1) for x in ll] + [0 for x in range(self.sequence_length - len(ll))]
                    else:
                        feature += [(int(x) + add_len + 1) for x in ll[:self.sequence_length]]
                data_feature.append(feature)
                '''
        return data_label, data_feature


    def load_dic_len(self, path):
        dic_len = 0;
        with open(path) as f:
            for line in f:
                dic_len += 1
        return dic_len

    def load_transfer(self, path, add_len):
        res = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                index_en, index_cn, _, _ = line.split("\t")
                if self.cn2en == 1:
                    res[int(index_en) + 1] = int(index_cn) + 1 + add_len
                else:
                    res[int(index_cn) + 1] = int(index_en) + 1 + add_len
        return res

    def shuffle_data(self, label, feature, part = 1):
        assert(len(label) == len(feature))
        shuffle_indices = np.random.permutation(np.arange(len(label)))
        feature_shuffled = [feature[i] for i in shuffle_indices]
        label_shuffled = [label[i] for i in shuffle_indices]
        if part == 1:
            return feature_shuffled, label_shuffled
        train = int(len(label) * 1.0 / part)
        return feature_shuffled[:train], feature_shuffled[train:], label_shuffled[:train], label_shuffled[train:]

    def batch_iter(self, data, batch_size, shuffle = True):
        data_size = len(data)
        num_batches = int(len(data)/batch_size) + 1
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = [data[x] for x in shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index < end_index:
                yield shuffled_data[start_index:end_index]

    def load_cross_data(self):
        print("Loading data...")
        target_dic_len = self.load_dic_len(self.target_dic_path)
        print("target dic " + self.target_dic_path + " len: " + str(target_dic_len))
        source_dic_len = self.load_dic_len(self.source_dic_path)
        print("source dic " + self.source_dic_path + " len: " + str(source_dic_len))
        self.vocab_size = target_dic_len + source_dic_len + 1
        
        target_label, target_feature = self.load_data(self.target_path, source_dic_len)
        print("target_path: " + self.target_path + " || len: " + str(len(target_label)))
        source_label, source_feature = self.load_data(self.source_path)
        print("source_path: " + self.source_path + " || len: " + str(len(source_label)))

        self.transform_dic = self.load_transfer(self.transfer_path, source_dic_len)
        
        target_train_feature, self.target_test_feature, target_train_label, self.target_test_label = \
            self.shuffle_data(target_label, target_feature, self.part)
        source_train_feature, source_train_label = self.shuffle_data(source_label, source_feature)
        
        print("Target Train/Test split: {:d}/{:d}".format(len(target_train_feature), len(self.target_test_feature)))
        print("Source: {:d}".format(len(source_train_feature)))

        self.all_batches = []
        for i in range(self.num_epochs):
            batches = []
            for j in range(self.part):
                arr_1 = [1 for k in range(len(target_train_label))]
                arr_0 = [0 for k in range(len(target_train_label))]
                target_batches = self.batch_iter(list(zip(target_train_label, target_train_feature, arr_1, arr_0)), self.batch_size)
                batches += target_batches
            source_batches = self.batch_iter(list(zip(source_train_label, source_train_feature, arr_1, arr_0)), self.batch_size)
            batches += source_batches
            random.shuffle(batches)
            self.all_batches += batches

    def cross_training(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                if self.model == "cnn":
                    cross_model = cross_cnn(
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda)
                else:
                    pass

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cross_model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    
                sess.run(tf.global_variables_initializer())

                #training and test
                for batch in self.all_batches:
                    y_batch, x_batch, f_cn, f_en = zip(*batch)
                    trans_en = set()
                    trans_cn = []
                    if f_en == 1:
                        for tt in x_batch:
                            for ttt in tt:
                                if ttt in self.transform_dic.keys():
                                    trans_en.add(ttt)
                        trans_en = list(trans_en)
                        trans_cn = [self.transform_dic[te] for te in trans_en]
                    else:
                        trans_en = [0]
                        trans_cn = [0]
                    
                    # train 
                    feed_dict = {
                        cross_model.input_f_en: f_en[0],
                        cross_model.input_f_cn: f_cn[0],
                        cross_model.input_trans_en: trans_en,
                        cross_model.input_trans_cn: trans_cn,
                        cross_model.input_x: x_batch,
                        cross_model.input_y: y_batch,
                        cross_model.dropout_keep_prob: self.dropout_keep_prob,
                        cross_model.en_weight: 0.1,
                        cross_model.trans_weight: 0.1
                    }
                    _, step, loss, kl, accuracy = sess.run(
                        [train_op, global_step, cross_model.loss, cross_model.kl, cross_model.accuracy],
                        feed_dict)
                    time_str = datetime.now().isoformat()
                    #print("train {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))

                    # test
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        feed_dict = {
                            cross_model.input_f_en: 0,
                            cross_model.input_f_cn: 1,
                            cross_model.input_trans_en: [0],
                            cross_model.input_trans_cn: [0],
                            cross_model.input_x: self.target_test_feature,
                            cross_model.input_y: self.target_test_label,
                            cross_model.dropout_keep_prob: 1,
                            cross_model.en_weight: 0,
                            cross_model.trans_weight: 0
                        }
                        step, loss, kl, accuracy = sess.run(
                            [global_step, cross_model.loss, cross_model.kl, cross_model.accuracy],
                            feed_dict)
                        time_str = datetime.now().isoformat()
                        print("eval  {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))


    def load_single_data(self):
        print("Loading data...")
        target_dic_len = self.load_dic_len(self.target_dic_path)
        print("target dic " + self.target_dic_path + " len: " + str(target_dic_len))
        self.vocab_size = target_dic_len + 1
        target_label, target_feature = self.load_data(self.target_path)

        target_train_feature, self.target_test_feature, target_train_label, self.target_test_label = \
            self.shuffle_data(target_label, target_feature, self.part)

        print("Target Train/Test split: {:d}/{:d}".format(len(target_train_feature), len(self.target_test_feature)))

        self.all_batches = []
        for i in range(self.num_epochs):
            batches = []
            for j in range(self.part):
                target_batches = self.batch_iter(list(zip(target_train_label, target_train_feature)), self.batch_size)
                batches += target_batches
            random.shuffle(batches)
            self.all_batches += batches
    

    def single_training(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)
            sess = tf.Session(config = session_conf)
            with sess.as_default():
                if self.model == "cnn":
                    single_model = single_cnn(
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda)
                elif self.model == "bi_lstm":
                    single_model = single_bi_lstm(
                        sequence_length = self.sequence_length,
                        num_classes = len(self.emotion_list),
                        vocab_size = self.vocab_size,
                        embedding_size = self.embedding_dim,
                        filter_sizes = self.filter_sizes,
                        num_filters = self.num_filters,
                        l2_reg_lambda = self.l2_reg_lambda,
                        dropout_keep_prob = 0.5)
                else:
                    pass
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(single_model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    
                sess.run(tf.global_variables_initializer())

                def gene_pad_seq(batches, seq_len):
                    batch_seq_len = []
                    batch_pad = []
                    for b in batches:
                        if len(b) < seq_len:
                            feature = b + [0 for i in range(seq_len - len(b))]
                            batch_seq_len.append(len(b))
                        else:
                            feature = b[:seq_len]
                            batch_seq_len.append(seq_len)
                        batch_pad.append(feature)
                    return batch_pad, batch_seq_len

                #training and test
                for batch in self.all_batches:
                    y_batch, x_batch = zip(*batch)
                    
                    x_batch_pad, x_batch_seq_len = gene_pad_seq(x_batch, self.sequence_length) 
                    feed_dict = {
                        single_model.input_x: x_batch_pad,
                        single_model.input_y: y_batch,
                        single_model.seq_len: x_batch_seq_len,
                        single_model.dropout_keep_prob: self.dropout_keep_prob,
                    }
                    _, step, loss, kl, accuracy = sess.run(
                        [train_op, global_step, single_model.loss, single_model.kl, single_model.accuracy],
                        feed_dict)
                    time_str = datetime.now().isoformat()
                    #print("train {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))

                    # test
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.evaluate_every == 0:
                        target_test_feature_pad, target_test_feature_seq_len = gene_pad_seq(self.target_test_feature, self.sequence_length)
                        feed_dict = {
                            single_model.input_x: target_test_feature_pad,
                            single_model.input_y: self.target_test_label,
                            single_model.seq_len: target_test_feature_seq_len,
                            single_model.dropout_keep_prob: 1,
                        }
                        step, loss, kl, accuracy = sess.run(
                            [global_step, single_model.loss, single_model.kl, single_model.accuracy],
                            feed_dict)
                        time_str = datetime.now().isoformat()
                        print("eval  {}: step {}, loss {:g}, kl {:g}, acc {:g}".format(time_str, step, loss, kl, accuracy))

        
