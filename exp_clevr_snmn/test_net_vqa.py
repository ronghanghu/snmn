import argparse
import os
import numpy as np
import tensorflow as tf

from models_clevr_snmn.model import Model
from models_clevr_snmn.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.clevr_train.data_reader import DataReader

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
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
data_reader = DataReader(
    imdb_file, shuffle=False, one_pass=True, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)
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
    result_dir, 'vqa_%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
os.makedirs(result_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Run test
answer_correct, num_questions = 0, 0
if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
    output_predictions = []
    answer_word_list = data_reader.batch_loader.answer_dict.word_list
    pred_file = os.path.join(
        result_dir, 'vqa_eval_preds_%s_%s_%08d.txt' % (
            cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, cfg.TEST.ITER))
for n_batch, batch in enumerate(data_reader.batches()):
    if 'answer_label_batch' not in batch:
        batch['answer_label_batch'] = -np.ones(
            len(batch['image_feat_batch']), np.int32)
        if num_questions == 0:
            print('imdb has no answer labels. Using dummy labels.\n\n'
                  '**The final accuracy will be zero (no labels provided)**\n')

    fetch_list = [model.vqa_scores]
    answer_incorrect = num_questions - answer_correct
    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
        run_vis = (
            answer_correct < cfg.TEST.NUM_VIS_CORRECT or
            answer_incorrect < cfg.TEST.NUM_VIS_INCORRECT)
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
        model.vis_batch_vqa(
            data_reader, batch, fetch_list_val[-1], num_questions,
            answer_correct, answer_incorrect, vis_dir)

    # compute accuracy
    vqa_scores_val = fetch_list_val[0]
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)
    answer_correct += np.sum(vqa_predictions == vqa_labels)
    num_questions += len(vqa_labels)
    accuracy = answer_correct / num_questions
    if n_batch % 20 == 0:
        print('exp: %s, iter = %d, accumulated accuracy on %s = %f (%d / %d)' %
              (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
               accuracy, answer_correct, num_questions))

    if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
        output_predictions += [answer_word_list[p] for p in vqa_predictions]

with open(os.path.join(
        result_dir, 'vqa_results_%s.txt' % cfg.TEST.SPLIT_VQA), 'w') as f:
    print('\nexp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions))
    print('exp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions), file=f)

if cfg.TEST.OUTPUT_VQA_EVAL_PRED:
    with open(pred_file, 'w') as f:
        f.writelines([a + '\n' for a in output_predictions])
    print('prediction file written to %s' % pred_file)
