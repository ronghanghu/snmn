import argparse
import os
import numpy as np
import tensorflow as tf

from models_clevr_snmn.model import Model
from models_clevr_snmn.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.clevr_train.data_reader import DataReader
from util import boxes
from util.losses import SharpenLossScaler

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
if args.opts:
    merge_cfg_from_list(args.opts)


# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_LOC
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    img_H=cfg.MODEL.H_IMG, img_W=cfg.MODEL.W_IMG)
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_choices = data_reader.batch_loader.answer_dict.num_vocab
module_names = data_reader.batch_loader.layout_dict.word_list

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
model = Model(
    input_seq_batch, seq_length_batch, image_feat_batch, num_vocab=num_vocab,
    num_choices=num_choices, module_names=module_names, is_training=True)

# Loss function
bbox_ind_batch = tf.placeholder(tf.int32, [None])
bbox_offset_batch = tf.placeholder(tf.float32, [None, 4])
loss_bbox_ind = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.loc_scores, labels=bbox_ind_batch))
loss_bbox_offset = model.bbox_offset_loss(bbox_ind_batch, bbox_offset_batch)
if cfg.TRAIN.USE_GT_LAYOUT:
    gt_layout_batch = tf.placeholder(tf.int32, [None, None])
    loss_layout = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.module_logits, labels=gt_layout_batch))
else:
    loss_layout = tf.convert_to_tensor(0.)
loss_rec = model.rec_loss
sharpen_scale_ph = tf.placeholder(tf.float32, [])
if cfg.TRAIN.USE_SHARPEN_LOSS:
    loss_sharpen = model.sharpen_loss()
else:
    loss_sharpen = tf.convert_to_tensor(0.)
loss_train = (loss_bbox_ind * cfg.TRAIN.BBOX_IND_LOSS_WEIGHT +
              loss_bbox_offset * cfg.TRAIN.BBOX_OFFSET_LOSS_WEIGHT +
              loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
              loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT +
              loss_sharpen * cfg.TRAIN.SHARPEN_LOSS_WEIGHT * sharpen_scale_ph)
loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
solver_op = solver.minimize(loss_total)
# Save moving average of parameters
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
ema_op = ema.apply(model.params)
with tf.control_dependencies([solver_op]):
    train_op = tf.group(ema_op)

# Save snapshot
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
if cfg.TRAIN.START_ITER > 0:
    snapshot_file = os.path.join(snapshot_dir, "%08d" % cfg.TRAIN.START_ITER)
    print('resume training from %s' % snapshot_file)
    snapshot_saver.restore(sess, snapshot_file)
else:
    sess.run(tf.global_variables_initializer())
    if cfg.TRAIN.INIT_FROM_WEIGHTS:
        snapshot_saver.restore(sess, cfg.TRAIN.INIT_WEIGHTS_FILE)
        print('initialized from %s' % cfg.TRAIN.INIT_WEIGHTS_FILE)
# Save config
np.save(os.path.join(snapshot_dir, 'cfg.npy'), np.array(cfg))

# Write summary to TensorBoard
log_dir = cfg.TRAIN.LOG_DIR % cfg.EXP_NAME
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_bbox_ind_ph = tf.placeholder(tf.float32, [])
loss_bbox_offset_ph = tf.placeholder(tf.float32, [])
loss_layout_ph = tf.placeholder(tf.float32, [])
loss_rec_ph = tf.placeholder(tf.float32, [])
loss_sharpen_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("loss/bbox_ind", loss_bbox_ind_ph))
summary_trn.append(tf.summary.scalar("loss/bbox_offset", loss_bbox_offset_ph))
summary_trn.append(tf.summary.scalar("loss/layout", loss_layout_ph))
summary_trn.append(tf.summary.scalar("loss/rec", loss_rec_ph))
summary_trn.append(tf.summary.scalar("loss/sharpen", loss_sharpen_ph))
summary_trn.append(tf.summary.scalar("loss/sharpen_scale", sharpen_scale_ph))
summary_trn.append(tf.summary.scalar("eval/loc/P1", accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
iou_th = cfg.TRAIN.BBOX_IOU_THRESH
sharpen_loss_scaler = SharpenLossScaler(cfg)
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_ITER
    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    sharpen_scale = sharpen_loss_scaler(n_iter)
    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch'],
                 bbox_ind_batch: batch['bbox_ind_batch'],
                 bbox_offset_batch: batch['bbox_offset_batch'],
                 sharpen_scale_ph: sharpen_scale}
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch['gt_layout_batch']
    loc_scores_val, bbox_offset_val, loss_bbox_ind_val, loss_bbox_offset_val, \
        loss_layout_val, loss_rec_val, loss_sharpen_val, _ = sess.run(
            (model.loc_scores, model.bbox_offset, loss_bbox_ind,
             loss_bbox_offset, loss_layout, loss_rec, loss_sharpen, train_op),
            feed_dict)

    # compute accuracy
    bbox_pred = boxes.batch_feat_grid2bbox(
        np.argmax(loc_scores_val, axis=1), bbox_offset_val,
        data_reader.batch_loader.stride_H, data_reader.batch_loader.stride_W,
        data_reader.batch_loader.feat_H, data_reader.batch_loader.feat_W)
    bbox_gt = batch['bbox_batch']
    accuracy = np.mean(boxes.batch_bbox_iou(bbox_pred, bbox_gt) >= iou_th)
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # Add to TensorBoard summary
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        print("exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (bbox_ind) = %f, loss (bbox_offset) = %f, "
              "loss (layout) = %f, loss (rec) = %f, "
              "loss (sharpen) = %f, sharpen_scale = %f\n\t" % (
                loss_bbox_ind_val, loss_bbox_offset_val, loss_layout_val,
                loss_rec_val, loss_sharpen_val, sharpen_scale) +
              "P1@%.2f (cur) = %f, P1@%.2f (avg) = %f" % (
                iou_th, accuracy, iou_th, avg_accuracy))
        summary = sess.run(log_step_trn, {
            loss_bbox_ind_ph: loss_bbox_ind_val,
            loss_bbox_offset_ph: loss_bbox_offset_val,
            loss_layout_ph: loss_layout_val,
            loss_rec_ph: loss_rec_val,
            loss_sharpen_ph: loss_sharpen_val, sharpen_scale_ph: sharpen_scale,
            accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)

    # Save snapshot
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
