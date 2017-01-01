__doc__ = """RNN-based GAN.  For applying Generative Adversarial Networks to sequential data."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


def _cumsum(x, length):
    lower_triangular_ones = tf.constant(
        np.tril(np.ones((length, length))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(lower_triangular_ones,
                      tf.reshape(x, [length, 1])),
            [length])


def _backwards_cumsum(x, length):
    upper_triangular_ones = tf.constant(
        np.triu(np.ones((length, length))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(upper_triangular_ones,
                      tf.reshape(x, [length, 1])),
            [length])


class RNN(object):

    def __init__(self, num_emb, emb_dim, hidden_dim,
                 sequence_length, start_token,
                 learning_rate=0.01, reward_gamma=0.9):
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant(start_token, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params, self.g_embeddings)  # maps h_t to o_t (output token logits)

        with tf.variable_scope('discriminator'):
            self.d_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.d_params.append(self.d_embeddings)
            self.d_recurrent_unit = self.create_recurrent_unit(self.d_params)  # maps h_tm1 to h_t for discriminator
            self.d_classifier_unit = self.create_classifier_unit(self.d_params)  # maps h_t to class prediction logits
            self.d_h0 = tf.Variable(self.init_vector([self.hidden_dim]))
            self.d_params.append(self.d_h0)

        self.h0 = tf.placeholder(tf.float32, shape=[self.hidden_dim])  # initial random vector for generator
        self.x = tf.placeholder(tf.int32, shape=[self.sequence_length])  # sequence of indices of true data, not including start token
        self.samples = tf.placeholder(tf.float32, shape=[self.sequence_length])  # random samples from [0, 1]

        # generator on initial randomness
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        samples = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        samples = samples.unpack(self.samples)
        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            sample = samples.read(i)
            o_cumsum = _cumsum(o_t, self.num_emb)  # prepare for sampling
            next_token = tf.to_int32(tf.reduce_min(tf.where(sample < o_cumsum)))  # sample
            x_tp1 = tf.gather(self.g_embeddings, next_token)
            gen_o = gen_o.write(i, tf.gather(o_t, next_token))  # we only need the sampled token's probability
            gen_x = gen_x.write(i, next_token)  # indices, not embeddings
            return i + 1, x_tp1, h_t, gen_o, gen_x

        _, _, _, self.gen_o, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.gather(self.g_embeddings, self.start_token),
                       self.h0, gen_o, gen_x))

        # discriminator on generated and real data
        d_gen_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        d_real_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        self.gen_x = self.gen_x.pack()
        emb_gen_x = tf.gather(self.d_embeddings, self.gen_x)
        ta_emb_gen_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_gen_x = ta_emb_gen_x.unpack(emb_gen_x)

        emb_real_x = tf.gather(self.d_embeddings, self.x)
        ta_emb_real_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_real_x = ta_emb_real_x.unpack(emb_real_x)

        def _d_recurrence(i, inputs, h_tm1, pred):
            x_t = inputs.read(i)
            h_t = self.d_recurrent_unit(x_t, h_tm1)
            y_t = self.d_classifier_unit(h_t)
            pred = pred.write(i, y_t)
            return i + 1, inputs, h_t, pred

        _, _, _, self.d_gen_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       ta_emb_gen_x,
                       self.d_h0,
                       d_gen_predictions))
        self.d_gen_predictions = tf.reshape(
                self.d_gen_predictions.pack(),
                [self.sequence_length])

        _, _, _, self.d_real_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       ta_emb_real_x,
                       self.d_h0,
                       d_real_predictions))
        self.d_real_predictions = tf.reshape(
                self.d_real_predictions.pack(),
                [self.sequence_length])

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        emb_x = tf.gather(self.g_embeddings, self.x)
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unpack(emb_x)

        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            o_t = self.g_output_unit(h_t)
            g_predictions = g_predictions.write(i, o_t)
            x_tp1 = ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        _, _, _, self.g_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.gather(self.g_embeddings, self.start_token),
                       self.h0, g_predictions))

        self.g_predictions = tf.reshape(
                self.g_predictions.pack(),
                [self.sequence_length, self.num_emb])

        # calculate discriminator loss
        self.d_gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_gen_predictions, tf.zeros([self.sequence_length])))
        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_real_predictions, tf.ones([self.sequence_length])))

        # calculate generator rewards and loss
        decays = tf.exp(tf.log(self.reward_gamma) * tf.to_float(tf.range(self.sequence_length)))
        rewards = _backwards_cumsum(decays * tf.sigmoid(self.d_gen_predictions),
                                    self.sequence_length)
        normalized_rewards = \
            rewards / _backwards_cumsum(decays, self.sequence_length) - self.expected_reward

        self.reward_loss = tf.reduce_mean(normalized_rewards ** 2)
        self.g_loss = \
            -tf.reduce_mean(tf.log(self.gen_o.pack()) * normalized_rewards)

        # pretraining loss
        self.pretrain_loss = \
            (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x),
                           self.num_emb, 1.0, 0.0) * tf.log(self.g_predictions))
             / self.sequence_length)

        # training updates
        d_opt = self.d_optimizer(self.learning_rate)
        g_opt = self.g_optimizer(self.learning_rate)
        pretrain_opt = self.g_optimizer(self.learning_rate)
        reward_opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.d_gen_grad = tf.gradients(self.d_gen_loss, self.d_params)
        self.d_real_grad = tf.gradients(self.d_real_loss, self.d_params)
        self.d_gen_updates = d_opt.apply_gradients(zip(self.d_gen_grad, self.d_params))
        self.d_real_updates = d_opt.apply_gradients(zip(self.d_real_grad, self.d_params))

        self.reward_grad = tf.gradients(self.reward_loss, [self.expected_reward])
        self.reward_updates = reward_opt.apply_gradients(zip(self.reward_grad, [self.expected_reward]))

        self.g_grad = tf.gradients(self.g_loss, self.g_params)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

        self.pretrain_grad = tf.gradients(self.pretrain_loss, self.g_params)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))

    def generate(self, session):
        outputs = session.run(
                [self.gen_x],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs[0]

    def train_g_step(self, session):
        outputs = session.run(
                [self.g_updates, self.reward_updates, self.g_loss,
                 self.expected_reward, self.gen_x],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs

    def train_d_gen_step(self, session):
        outputs = session.run(
                [self.d_gen_updates, self.d_gen_loss],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs

    def train_d_real_step(self, session, x):
        outputs = session.run([self.d_real_updates, self.d_real_loss],
                              feed_dict={self.x: x})
        return outputs

    def pretrain_step(self, session, x):
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions],
                              feed_dict={self.x: x,
                                         self.h0: np.random.normal(size=self.hidden_dim)})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)

    def create_recurrent_unit(self, params):
        self.W_rec = tf.Variable(self.init_matrix([self.hidden_dim, self.emb_dim]))
        params.append(self.W_rec)
        def unit(x_t, h_tm1):
            return h_tm1 + tf.reshape(tf.matmul(self.W_rec, tf.reshape(x_t, [self.emb_dim, 1])), [self.hidden_dim])
        return unit

    def create_output_unit(self, params, embeddings):
        self.W_out = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.b_out1 = tf.Variable(self.init_vector([self.emb_dim, 1]))
        self.b_out2 = tf.Variable(self.init_vector([self.num_emb, 1]))
        params.extend([self.W_out, self.b_out1, self.b_out2])
        def unit(h_t):
            logits = tf.reshape(
                    self.b_out2 +
                    tf.matmul(embeddings,
                              tf.tanh(self.b_out1 +
                                      tf.matmul(self.W_out, tf.reshape(h_t, [self.hidden_dim, 1])))),
                    [1, self.num_emb])
            return tf.reshape(tf.nn.softmax(logits), [self.num_emb])
        return unit

    def create_classifier_unit(self, params):
        self.W_class = tf.Variable(self.init_matrix([1, self.hidden_dim]))
        self.b_class = tf.Variable(self.init_vector([1]))
        params.extend([self.W_class, self.b_class])
        def unit(h_t):
            return self.b_class + tf.matmul(self.W_class, tf.reshape(h_t, [self.hidden_dim, 1]))
        return unit

    def d_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)


class GRU(RNN):

    def create_recurrent_unit(self, params):
        self.W_rx = tf.Variable(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.W_zx = tf.Variable(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.W_hx = tf.Variable(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_rh = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_zh = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_hh = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        params.extend([
            self.W_rx, self.W_zx, self.W_hx,
            self.U_rh, self.U_zh, self.U_hh])

        def unit(x_t, h_tm1):
            x_t = tf.reshape(x_t, [self.emb_dim, 1])
            h_tm1 = tf.reshape(h_tm1, [self.hidden_dim, 1])
            r = tf.sigmoid(tf.matmul(self.W_rx, x_t) + tf.matmul(self.U_rh, h_tm1))
            z = tf.sigmoid(tf.matmul(self.W_zx, x_t) + tf.matmul(self.U_zh, h_tm1))
            h_tilda = tf.tanh(tf.matmul(self.W_hx, x_t) + tf.matmul(self.U_hh, r * h_tm1))
            h_t = (1 - z) * h_tm1 + z * h_tilda
            return tf.reshape(h_t, [self.hidden_dim])

        return unit
