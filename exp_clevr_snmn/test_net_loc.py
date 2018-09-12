import argparse
import os
import numpy as np
import tensorflow as tf

from models_clevr_snmn.model import Model
from models_clevr_snmn.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.clevr_train.data_reader import DataReader
from util import boxes

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
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_LOC
data_reader = DataReader(
    imdb_file, shuffle=False, one_pass=True, batch_size=cfg.TRAIN.BATCH_SIZE,
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
    num_choices=num_choices, module_names=module_names, is_training=False)

# Load snapshot
if cfg.TEST.USE_EMV:
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
    var_names = {
        (ema.average_name(v) if v in model.params else v.op.name): v
        for v in tf.global_variables()}
else:
    var_names = {v.op.name: v for v in tf.global_variables()}
snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.ITER)
snapshot_saver = tf.train.Saver(var_names)
snapshot_saver.restore(sess, snapshot_file)

# Write results
result_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.ITER)
vis_dir = os.path.join(
    result_dir, 'loc_%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_LOC))
os.makedirs(result_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Run test
bbox_correct, num_questions = 0, 0
iou_th = cfg.TEST.BBOX_IOU_THRESH
for n_batch, batch in enumerate(data_reader.batches()):
    fetch_list = [model.loc_scores, model.bbox_offset]
    bbox_incorrect = num_questions - bbox_correct
    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
        run_vis = (
            bbox_correct < cfg.TEST.NUM_VIS_CORRECT or
            bbox_incorrect < cfg.TEST.NUM_VIS_INCORRECT)
    else:
        run_vis = num_questions < cfg.TEST.NUM_VIS
    if run_vis:
        fetch_list.append(model.vis_outputs)
    fetch_list_val = sess.run(fetch_list, feed_dict={
            input_seq_batch: batch['input_seq_batch'],
            seq_length_batch: batch['seq_length_batch'],
            image_feat_batch: batch['image_feat_batch']})

    # visualization
    if run_vis:
        model.vis_batch_loc(
            data_reader, batch, fetch_list_val[-1], num_questions,
            bbox_correct, bbox_incorrect, vis_dir)

    # compute accuracy
    loc_scores_val, bbox_offset_val = fetch_list_val[0:2]
    bbox_pred = boxes.batch_feat_grid2bbox(
        np.argmax(loc_scores_val, axis=1), bbox_offset_val,
        data_reader.batch_loader.stride_H, data_reader.batch_loader.stride_W,
        data_reader.batch_loader.feat_H, data_reader.batch_loader.feat_W)
    bbox_gt = batch['bbox_batch']
    bbox_correct += np.sum(boxes.batch_bbox_iou(bbox_pred, bbox_gt) >= iou_th)
    num_questions += len(bbox_gt)
    accuracy = bbox_correct / num_questions
    if n_batch % 20 == 0:
        print('exp: %s, iter = %d, accumulated P1@%.2f on %s = %f (%d / %d)' %
              (cfg.EXP_NAME, cfg.TEST.ITER, iou_th, cfg.TEST.SPLIT_LOC,
               accuracy, bbox_correct, num_questions))

with open(os.path.join(
        result_dir, 'loc_results_%s.txt' % cfg.TEST.SPLIT_LOC), 'w') as f:
    print('\nexp: %s, iter = %d, final P1@%.2f on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, iou_th, cfg.TEST.SPLIT_LOC, accuracy,
           bbox_correct, num_questions))
    print('exp: %s, iter = %d, final P1@%.2f on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, iou_th, cfg.TEST.SPLIT_LOC, accuracy,
           bbox_correct, num_questions), file=f)
