import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from .config import cfg
from util.cnn import fc_layer as fc, fc_elu_layer as fc_elu
from util.gumbel_softmax import gumbel_softmax


class Controller:

    def __init__(self, lstm_seq, q_encoding, embed_seq, seq_length_batch,
                 num_module, scope='controller', reuse=None):
        """
        Build the controller that is used to give inputs to the neural modules.
        The controller unrolls itself for a fixed number of time steps.
        All parameters are shared across time steps.

        # The controller uses an auto-regressive structure like a GRU cell.
        # Attention is used over the input sequence.
        Here, the controller is the same as in the previous MAC paper, and
        additional module weights

        Input:
            lstm_seq: [S, N, d], tf.float32
            q_encoding: [N, d], tf.float32
            embed_seq: [S, N, e], tf.float32
            seq_length_batch: [N], tf.int32
        """

        dim = cfg.MODEL.LSTM_DIM
        ctrl_dim = (cfg.MODEL.EMBED_DIM if cfg.MODEL.CTRL.USE_WORD_EMBED
                    else cfg.MODEL.LSTM_DIM)
        T_ctrl = cfg.MODEL.T_CTRL

        # an attention mask to normalize textual attention over the actual
        # sequence length
        S = tf.shape(lstm_seq)[0]
        N = tf.shape(lstm_seq)[1]
        # att_mask: [S, N, 1]
        att_mask = tf.less(tf.range(S)[:, ax, ax], seq_length_batch[:, ax])
        att_mask = tf.cast(att_mask, tf.float32)
        with tf.variable_scope(scope, reuse=reuse):
            S = tf.shape(lstm_seq)[0]
            N = tf.shape(lstm_seq)[1]

            # manually unrolling for a number of timesteps
            c_init = tf.get_variable(
                'c_init', [1, ctrl_dim],
                initializer=tf.initializers.random_normal(
                    stddev=np.sqrt(1. / ctrl_dim)))
            c_prev = tf.tile(c_init, to_T([N, 1]))
            c_prev.set_shape([None, ctrl_dim])
            c_list = []
            cv_list = []
            module_logit_list = []
            module_prob_list = []
            for t in range(T_ctrl):
                q_i = fc('fc_q_%d' % t, q_encoding, output_dim=dim)  # [N, d]
                q_i_c = tf.concat([q_i, c_prev], axis=1)  # [N, 2d]
                cq_i = fc('fc_cq', q_i_c, output_dim=dim, reuse=(t > 0))

                # Apply a fully connected network on top of cq_i to predict the
                # module weights
                module_w_l1 = fc_elu(
                    'fc_module_w_layer1', cq_i, output_dim=dim, reuse=(t > 0))
                module_w_l2 = fc(
                    'fc_module_w_layer2', module_w_l1, output_dim=num_module,
                    reuse=(t > 0))  # [N, M]
                module_logit_list.append(module_w_l2)
                if cfg.MODEL.CTRL.USE_GUMBEL_SOFTMAX:
                    module_prob = gumbel_softmax(
                        module_w_l2, cfg.MODEL.CTRL.GUMBEL_SOFTMAX_TMP)
                else:
                    module_prob = tf.nn.softmax(module_w_l2, axis=1)
                module_prob_list.append(module_prob)

                elem_prod = tf.reshape(cq_i * lstm_seq, to_T([S*N, dim]))
                elem_prod.set_shape([None, dim])  # [S*N, d]
                raw_cv_i = tf.reshape(
                    fc('fc_cv_i', elem_prod, output_dim=1, reuse=(t > 0)),
                    to_T([S, N, 1]))
                cv_i = tf.nn.softmax(raw_cv_i, axis=0)  # [S, N, 1]
                # normalize the attention over the actual sequence length
                if cfg.MODEL.CTRL.NORMALIZE_ATT:
                    cv_i = cv_i * att_mask
                    cv_i /= tf.reduce_sum(cv_i, 0, keepdims=True)

                if cfg.MODEL.CTRL.USE_WORD_EMBED:
                    c_i = tf.reduce_sum(cv_i * embed_seq, axis=[0])  # [N, e]
                else:
                    c_i = tf.reduce_sum(cv_i * lstm_seq, axis=[0])  # [N, d]
                c_list.append(c_i)
                cv_list.append(cv_i)
                c_prev = c_i

        self.module_logits = tf.stack(module_logit_list)
        self.module_probs = tf.stack(module_prob_list)
        self.module_prob_list = module_prob_list
        self.c_list = c_list
        self.cv_list = cv_list
