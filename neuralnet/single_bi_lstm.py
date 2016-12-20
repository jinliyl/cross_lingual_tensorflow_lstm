import tensorflow as tf
import numpy as np

class single_bi_lstm(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda = 0.0, label_smoothing = 0.1):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.seq_len = tf.placeholder(tf.int32, [None], name = "seq_len")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.label_smoothing = label_smoothing
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            self.embedded_chars_reshape = tf.reshape(self.embedded_chars, [-1, embedding_size])
            w = tf.Variable(tf.random_uniform([embedding_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0))
            b = tf.Variable(tf.random_uniform([embedding_size]))
            self.embedded_chars = tf.nn.xw_plus_b(embedded_chars_reshape, w, b)
            self.embedded_chars = tf.reshape(self.embedded_chars, [-1, sequence_length, embedding_size])

        # Lstm layer
        with tf.name_scope("lstm_layer"):
            # forward direction
            lstm_cell_f = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1)
            # backward direction
            lstm_cell_b = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1)
            lstm_cell_f = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_f, output_keep_prob = dropout_keep_prob)
            lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_b, output_keep_prob = dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_f, lstm_cell_b, self.embedded_chars, self.seq_len, dtype=tf.float32)
            #outputs, _ = tf.nn.dynamic_rnn(lstm_cell_f, self.embedded_chars, self.seq_len, dtype=tf.float32)
            output_fw, output_bw = outputs
            output_fw = tf.transpose(output_fw, [1, 0, 2])[-1]
            output_bw = tf.transpose(output_bw, [1, 0, 2])[-1]
            self.output_fb = tf.concat(1, [output_fw, output_bw])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output_fb, self.dropout_keep_prob)
            #self.h_drop = self.output_fb
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                shape=[embedding_size * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.log_softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.contrib.losses.softmax_cross_entropy(self.scores, self.input_y)
            selflosses = tf.contrib.losses.softmax_cross_entropy(tf.cast(self.input_y, tf.float32),
                        tf.cast(self.input_y, tf.float32))
            self.kl = tf.reduce_mean(losses - selflosses)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
