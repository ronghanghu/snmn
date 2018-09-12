import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from .config import cfg
from util.cnn import fc_layer as fc, fc_elu_layer as fc_elu, conv_layer as conv


def build_output_unit_vqa(q_encoding, m_last, num_choices, apply_dropout,
                          scope='output_unit', reuse=None):
    """
    Apply a 2-layer fully-connected network to predict answers. Apply dropout
    if specified.

    Input:
        q_encoding: [N, d], tf.float32
        m_last: [N, d], tf.float32
    Return:
        vqa_scores: [N, num_choices], tf.float32
    """

    output_dim = cfg.MODEL.VQA_OUTPUT_DIM
    with tf.variable_scope(scope, reuse=reuse):
        if cfg.MODEL.VQA_OUTPUT_USE_QUESTION:
            fc1 = fc_elu(
                'fc1', tf.concat([q_encoding, m_last], axis=1),
                output_dim=output_dim)
        else:
            fc1 = fc_elu('fc1_wo_q', m_last, output_dim=output_dim)
        if apply_dropout:
            fc1 = tf.nn.dropout(fc1, cfg.TRAIN.DROPOUT_KEEP_PROB)
        fc2 = fc('fc2', fc1, output_dim=num_choices)

        vqa_scores = fc2
    return vqa_scores


def build_output_unit_loc(q_encoding, kb_batch, att_last,
                          scope='output_unit_loc', reuse=None):
    """
    Apply a 1-layer convolution network to predict localization scores.
    Apply dropout
    if specified.

    Input:
        kb_batch: [N, H, W, d], tf.float32
        att_last: [N, H, W, 1], tf.float32
    Return:
        loc_scores: [N, H*W], tf.float32
        bbox_offset: [N, 4], tf.float32
    """

    with tf.variable_scope(scope, reuse=reuse):
        if cfg.MODEL.LOC_SCORES_POS_AFFINE:
            # make sure att signs do not flip
            w = tf.abs(tf.get_variable('loc_scores_affine_raw_w', []))
            b = tf.get_variable('loc_scores_affine_b', [])
            loc_scores = w * att_last + b
        else:
            loc_scores = conv(
                'conv_loc', att_last, kernel_size=3, stride=1, output_dim=1)
        loc_scores = tf.reshape(
            loc_scores, [-1, cfg.MODEL.H_FEAT*cfg.MODEL.W_FEAT])
        # extract the attended features for bounding box regression
        if cfg.MODEL.BBOX_REG_AS_FCN:
            if cfg.MODEL.BBOX_REG_USE_QUESTION:
                q_mapped = fc(
                    'fc_q_mapped', q_encoding, output_dim=cfg.MODEL.KB_DIM)
                bbox_offset_input = tf.nn.l2_normalize(
                    q_mapped[:, ax, ax, :] * kb_batch, axis=-1)
            else:
                bbox_offset_input = kb_batch
            bbox_offset_fcn = conv(
                'conv_bbox_offset', bbox_offset_input, 1, 1, output_dim=4)
            N = tf.shape(bbox_offset_fcn)[0]
            B = cfg.MODEL.H_FEAT*cfg.MODEL.W_FEAT  # B = H*W
            # bbox_offset_fcn [N, B, 4] is used for training
            bbox_offset_fcn = tf.reshape(bbox_offset_fcn, to_T([N, B, 4]))
            # bbox_offset [N, 4] is only used for prediction
            bbox_offset_flat = tf.reshape(bbox_offset_fcn, to_T([N*B, 4]))
            slice_inds = tf.range(N) * B + tf.argmax(
                loc_scores, axis=-1, output_type=tf.int32)
            bbox_offset = tf.gather(bbox_offset_flat, slice_inds)
        else:
            bbox_offset_fcn = None
            kb_loc = _extract_softmax_avg(kb_batch, att_last)
            if cfg.MODEL.BBOX_REG_USE_QUESTION:
                q_mapped = fc(
                    'fc_q_mapped', q_encoding, output_dim=cfg.MODEL.KB_DIM)
                elt_prod = tf.nn.l2_normalize(q_mapped * kb_loc, axis=-1)
                bbox_offset = fc(
                    'fc_bbox_offset_with_q', elt_prod, output_dim=4)
            else:
                bbox_offset = fc('fc_bbox_offset', kb_loc, output_dim=4)
    return loc_scores, bbox_offset, bbox_offset_fcn


def build_output_unit_rec(rec_inputs, input_seq_batch, embed_seq,
                          seq_length_batch, num_vocab, scope='output_unit_rec',
                          reuse=None):
    """
    Try to reconstruct the input sequence from the controller outputs with a
    seq-to-seq LSTM.

    Input:
        rec_inputs: [T, N, ?], tf.float32
        input_seq_batch: [S, N], tf.int32
        embed_seq: [S, N, e], tf.float32
        seq_length_batch: [N], tf.int32
    Return:
        loss_rec: [], tf.float32
    """
    with tf.variable_scope(scope, reuse=reuse):
        S = tf.shape(input_seq_batch)[0]
        N = tf.shape(input_seq_batch)[1]

        lstm_dim = cfg.MODEL.LSTM_DIM
        # encoder
        cell_encoder = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim, name='c_encoder')
        _, states_encoder = tf.nn.dynamic_rnn(
            cell_encoder, rec_inputs, dtype=tf.float32, time_major=True)
        # decoder
        cell_decoder = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim, name='c_decoder')
        embed_seq_shifted = tf.concat(
            [tf.zeros_like(embed_seq[:1]), embed_seq[:-1]], axis=0)
        outputs_decoder, _ = tf.nn.dynamic_rnn(
            cell_decoder, embed_seq_shifted, sequence_length=seq_length_batch,
            initial_state=states_encoder, time_major=True)

        # word prediction
        outputs_flat = tf.reshape(outputs_decoder, to_T([S*N, lstm_dim]))
        word_scores_flat = fc(
            'fc_word_scores', outputs_flat, output_dim=num_vocab)
        word_scores = tf.reshape(word_scores_flat, to_T([S, N, num_vocab]))

        # cross-entropy loss over the actual sequence words
        # att_mask: [S, N]
        att_mask = tf.less(tf.range(S)[:, ax], seq_length_batch)
        att_mask = tf.cast(att_mask, tf.float32)
        loss_rec = tf.reduce_sum(
            att_mask * tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=word_scores, labels=input_seq_batch)) / tf.reduce_sum(
                    att_mask)

    return loss_rec


def _spatial_softmax(att_raw):
    att_shape = tf.shape(att_raw)
    N = att_shape[0]
    att_softmax = tf.nn.softmax(tf.reshape(att_raw, to_T([N, -1])), axis=1)
    att_softmax = tf.reshape(att_softmax, att_shape)
    return att_softmax


def _extract_softmax_avg(kb_batch, att_raw):
    att_softmax = _spatial_softmax(att_raw)
    return tf.reduce_sum(kb_batch * att_softmax, axis=[1, 2])
