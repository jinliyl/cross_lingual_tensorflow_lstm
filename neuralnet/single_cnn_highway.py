import tensorflow as tf
import numpy as np


def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
  
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y

    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in range(layer_size):
       
        output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)

        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_

    return output


class single_cnn_highway(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda = 0.0, label_smoothing = 0.0, dropout_keep_prob = 0.5):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name = "input_y")
        self.seq_len = tf.placeholder(tf.int32, [None], name = "seq_len")
        self.batch_size = tf.placeholder(tf.int32, name = "batch_size")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
        self.label_smoothing = label_smoothing
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"), tf.variable_scope("CNN") as scope:
            self.embedded_W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedded_W, self.input_x)
            split_chars = tf.split(1, sequence_length, self.embedded_chars)
            #print(tf.shape(split_chars[0]))
            split_chars_highway = []
            #scope.reuse_variables()
            for idx in range(sequence_length):
                if idx != 0:
                    scope.reuse_variables()
                tmp = highway(tf.reshape(split_chars[idx], [-1, embedding_size]), embedding_size, layer_size = 2)
                split_chars_highway.append(tf.expand_dims(tmp, 0))
            split_chars_highway = tf.concat(0, split_chars_highway)
            split_chars_highway = tf.transpose(split_chars_highway, [1, 0, 2])
            self.embedded_chars_expanded = tf.expand_dims(split_chars_highway, -1)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
               
               

 
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            #self.h_drop = self.h_pool_flat
            # self.h_drop = highway(self.h_drop, self.h_drop.get_shape()[1], 4, 0.1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.log_softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.contrib.losses.softmax_cross_entropy(self.scores, self.input_y, label_smoothing = self.label_smoothing)
            selflosses = tf.contrib.losses.softmax_cross_entropy(tf.cast(self.input_y, tf.float32),
                        tf.cast(self.input_y, tf.float32), label_smoothing = self.label_smoothing)
            self.kl = tf.reduce_mean(losses - selflosses)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
