import matplotlib; matplotlib.use('Agg')  # NOQA

import os
import json
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

from .config import cfg
from util import boxes


def vis_one_vqa(img_path, words, vqa_scores, label, module_names, answers,
                txt_att, att_stack, stack_ptr, module_prob, save_path):
    img = skimage.io.imread(img_path)
    h = plt.figure(figsize=(20, 20))

    T = cfg.MODEL.T_CTRL

    # img
    plt.subplot(5, 3, 1)
    plt.imshow(img)
    plt.title(
        '\n'.join([' '.join(words[b:b+10]) for b in range(0, len(words), 10)]))

    # module weights
    plt.subplot(5, 3, 2)
    plt.imshow(module_prob.T, cmap='Reds')
    plt.colorbar()
    plt.xticks(range(T), range(T))
    plt.yticks(range(len(module_names)), module_names, size='small')
    plt.title('module weights at controller timestep')

    # textual attention
    plt.subplot(5, 3, 3)
    # print(np.sum(txt_att, axis=1))
    # print(np.sum(txt_att[:, :len(words)], axis=1))
    plt.imshow(txt_att[:, :len(words)], cmap='Reds')
    plt.colorbar()
    plt.xticks(range(len(words)), words, rotation=90)
    plt.yticks(range(T), range(T))
    plt.ylabel('controller timestep')
    plt.title('textual attention at controller timestep')

    # scores
    plt.subplot(5, 3, 4)
    plt.imshow(vqa_scores[np.newaxis, :], cmap='Reds')
    plt.xticks(range(len(answers)), answers, rotation=90)
    plt.yticks([], [])
    plt.xlabel('answer logits')
    plt.title('prediction: %s    label: %s' % (
        answers[np.argmax(vqa_scores)], answers[label]))

    plt.subplot(5, 3, 5)
    plt.imshow(stack_ptr.T, cmap='Reds')
    plt.colorbar()
    plt.xticks(range(T), range(T))
    plt.yticks(range(stack_ptr.shape[1]), range(stack_ptr.shape[1]))
    plt.ylabel('stack depth')
    plt.xlabel('stack pointer at controller timestep')

    # Visualize the attention stack
    # att_stack is T x H x W x L -> L x H x T x W
    plt.subplot(5, 3, 6)
    T, H, W, L = att_stack.shape
    plt.imshow(att_stack.transpose((3, 1, 0, 2)).reshape((L*H, T*W)))
    plt.colorbar()
    plt.xticks(W // 2 + np.arange(T) * W, range(T))
    plt.yticks(np.arange(L) * H, np.arange(L) * H)
    plt.ylabel('stack depth')
    plt.xlabel('image attention at controller timestep')

    # image attention at each timestep
    for t in range(T):
        plt.subplot(5, 3, t+7)
        att = np.sum(att_stack[t] * stack_ptr[t], axis=-1)
        img_with_att = attention_interpolation(img, att)
        plt.imshow(img_with_att)
        plt.xlabel('controller timestep t = %d' % t)

    plt.savefig(save_path)
    print('visualization saved to ' + save_path)
    plt.close(h)


def vis_one_loc(img_path, words, loc_scores, bbox_pred, bbox_gt, module_names,
                txt_att, att_stack, stack_ptr, module_prob, save_path):
    img = skimage.io.imread(img_path)
    h = plt.figure(figsize=(20, 20))

    T = cfg.MODEL.T_CTRL

    # img
    plt.subplot(5, 3, 1)
    plt.imshow(img)
    _print_bbox(bbox_pred, 'r')
    _print_bbox(bbox_gt, 'y')
    plt.title(
        '\n'.join([' '.join(words[b:b+10]) for b in range(0, len(words), 10)])
        + '\nred: prediction    yellow: ground-truth')

    # module weights
    plt.subplot(5, 3, 2)
    plt.imshow(module_prob.T, cmap='Reds')
    plt.colorbar()
    plt.xticks(range(T), range(T))
    plt.yticks(range(len(module_names)), module_names, size='small')
    plt.title('module weights at controller timestep')

    # textual attention
    plt.subplot(5, 3, 3)
    # print(np.sum(txt_att, axis=1))
    # print(np.sum(txt_att[:, :len(words)], axis=1))
    plt.imshow(txt_att[:, :len(words)], cmap='Reds')
    plt.colorbar()
    plt.xticks(range(len(words)), words, rotation=90)
    plt.yticks(range(T), range(T))
    plt.ylabel('controller timestep')
    plt.title('textual attention at controller timestep')

    # scores
    plt.subplot(5, 3, 4)
    plt.imshow(loc_scores.reshape(cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT))
    plt.colorbar()
    plt.title('localization scores')

    plt.subplot(5, 3, 5)
    plt.imshow(stack_ptr.T, cmap='Reds')
    plt.colorbar()
    plt.xticks(range(T), range(T))
    plt.yticks(range(stack_ptr.shape[1]), range(stack_ptr.shape[1]))
    plt.ylabel('stack depth')
    plt.xlabel('stack pointer at controller timestep')

    # Visualize the attention stack
    # att_stack is T x H x W x L -> L x H x T x W
    plt.subplot(5, 3, 6)
    T, H, W, L = att_stack.shape
    plt.imshow(att_stack.transpose((3, 1, 0, 2)).reshape((L*H, T*W)))
    plt.colorbar()
    plt.xticks(W // 2 + np.arange(T) * W, range(T))
    plt.yticks(np.arange(L) * H, np.arange(L) * H)
    plt.ylabel('stack depth')
    plt.xlabel('image attention at controller timestep')

    # image attention at each timestep
    for t in range(T):
        plt.subplot(5, 3, t+7)
        att = np.sum(att_stack[t] * stack_ptr[t], axis=-1)
        img_with_att = attention_interpolation(img, att)
        plt.imshow(img_with_att)
        plt.xlabel('controller timestep t = %d' % t)

    plt.savefig(save_path)
    print('visualization saved to ' + save_path)
    plt.close(h)


def _format_str(s):
    words = s.split()
    s = '\n'.join([' '.join(words[b:b+8]) for b in range(0, len(words), 8)])
    return s


MODULE_DESCRIPTION_TEXT = {
    '_NoOp':
        'it doesn\'t do anything (i.e. nothing is updated in this timestep).',  # NoQA
    '_Find':
        'it looks at new image regions based on attended text.',  # NoQA
    '_Transform':
        'it shifts the image attention to somewhere new, conditioned on its previous glimpse.',  # NoQA
    '_Filter':
        'it tries to select out some image regions from where it looked before (based on attended text).',  # NoQA
    '_And':
        'it takes the intersection of the program\'s two previous glimpses as inputs, returning their intersection.',  # NoQA
    '_Or':
        'it takes the union of the program\'s two previous glimpses as inputs, returning their union.',  # NoQA
    '_Scene':
        'it tries to look at some objects in the image.',  # NoQA
    '_DescribeOne':
        'it takes the program\'s previous glimpse as input, and tries to infer the answer from it.',  # NoQA
    '_DescribeTwo':
        'it takes the program\'s two previous glimpses as inputs, and tries to infer the answer from them.',  # NoQA

}


def _find_txt_segs(keep, words):
    segs = []
    elems = []
    for n, k in enumerate(keep):
        if k:
            elems.append(words[n])
        else:
            if elems:
                segs.append('"' + ' '.join(elems) + '"')
            elems = []
    if elems:
        segs.append('"' + ' '.join(elems) + '"')
    return segs


def _extract_txt_att(words, atts, thresh=0.5):
    """
    Take at most 3 words that have at least 50% of the max attention.
    """

    atts_sorted = np.sort(atts)[::-1]
    att_min = max(atts_sorted[2], atts_sorted[0]*thresh)
    # collect those words above att_min
    keep = (atts >= att_min)
    # assert np.any(keep)
    vis_txt = ', '.join(_find_txt_segs(keep, words))
    return vis_txt


def vis_one_stepwise(img_path, words, module_names, txt_att, att_stack,
                     stack_ptr, module_prob, save_path, vis_type,
                     vqa_scores=None, label=None, answers=None,
                     loc_scores=None, bbox_pred=None, bbox_gt=None):
    T = cfg.MODEL.T_CTRL
    # M = len(module_names)
    img = skimage.io.imread(img_path)
    scale_x = 480. / img.shape[1]
    scale_y = 320. / img.shape[0]
    img = skimage.transform.resize(img, (320, 480))

    h = plt.figure(figsize=(18, (T+2) * 5))
    if cfg.TEST.VIS_SHOW_IMG:
        # Image and question
        plt.subplot((T+2)*2, 3, (3, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('\n'.join(
            [' '.join(words[b:b+6]) for b in range(0, len(words), 6)]),
            fontsize=20)

    # Modules at each timestep
    m_list = [module_names[np.argmax(module_prob[t])] for t in range(T)]
    is_disp = np.ones(T, np.bool)
    is_ans = np.zeros(T, np.bool)
    if vis_type == 'vqa':
        """
        Show the output of the last "_Describe*"
        """
        describe_t = -1
        for t in range(T-1, -1, -1):
            if m_list[t].startswith('_Describe'):
                describe_t = t
                break
        for t in range(T):
            is_disp[t] = not (
                (m_list[t] == '_NoOp') or
                (m_list[t].startswith('_Describe') and t != describe_t))
        is_ans[describe_t] = True
    else:
        for t in range(T):
            is_disp[t] = (t == T-1) or not (
                (m_list[t] == '_NoOp') or
                (m_list[t].startswith('_Describe')))
        is_ans[T-1] = True

    t_disp = 0
    for t in range(T):
        if not is_disp[t]:
            continue
        show_ans = is_ans[t]

        m = m_list[t]
        if m in {'_Scene', '_NoOp', '_And', '_Or'}:
            att_txt = ''
        else:
            att_txt = _extract_txt_att(words, txt_att[t, :len(words)])

        if t == 0 and m == '_Filter':
            m_display = 'find'
        else:
            m_display = m[1:].replace(
                'Find', 'look_for').replace(
                'Filter', 'select').replace(
                'Transform', 'related_by').replace(
                'DescribeOne', 'Answer').replace(
                'DescribeTwo', 'Compare_Two').replace(
                'And', 'Intersect').replace('Or', 'Combine').lower()
        if show_ans and vis_type == 'loc' and \
                m in {'_NoOp', '_DescribeOne', '_DescribeTwo'}:
            m_display = 'bbox_regression'
            att_txt = ''

        # output attention
        if show_ans:
            if vis_type == 'vqa':
                plt.subplot((T+2)*2, 3, (6*t_disp+9, 6*t_disp+12))
                plt.imshow(np.ones(img.shape, np.float32))
                plt.axis('off')
                if cfg.TEST.VIS_SHOW_ANSWER:
                    answer_txt = (
                        'predicted answer: "%s"\ntrue answer: "%s"' % (
                            answers[np.argmax(vqa_scores)], answers[label]))
                else:
                    answer_txt = '(model prediction not shown)'
                plt.text(10, 100, answer_txt, fontsize=20)
            elif vis_type == 'loc':
                plt.subplot((T+2)*2, 3, (6*t_disp+9, 6*t_disp+12))
                plt.imshow(img)
                _print_bbox(bbox_gt, 'y', scale_x, scale_y)
                if cfg.TEST.VIS_SHOW_ANSWER:
                    _print_bbox(bbox_pred, 'r', scale_x, scale_y)
                    IoU = boxes.bbox_iou(bbox_pred, bbox_gt)
                    txt = 'prediction: red box\nground-truth: yellow box\n' \
                        '(IoU = %.2f)' % IoU
                else:
                    txt = 'prediction: (not shown)\nground-truth: yellow box'
                plt.xticks([], [])
                plt.yticks([], [])
                plt.xlabel(txt, fontsize=20)
            else:
                raise ValueError('Unknow vis_type ' + str(vis_type))
        else:
            plt.subplot((T+2)*2, 3, (6*t_disp+9, 6*t_disp+12))
            att = np.sum(att_stack[t] * stack_ptr[t], axis=-1)
            img_with_att = attention_interpolation(img, att)
            plt.imshow(img_with_att)
            plt.xticks([], [])
            plt.yticks([], [])
        plt.title('%s(%s)\n' % (m_display, att_txt), fontsize=24)
        patches = Arrow(
            img.shape[1] // 2, -35, 0, 32, width=40, color='k', clip_on=False)
        plt.gca().add_patch(patches)
        t_disp += 1

    plt.savefig(save_path, bbox_inches='tight')
    with open(save_path.replace('.png', '') + '.txt', 'w') as f:
        question = (' '.join(words)).replace(' ?', '?')
        if vis_type == 'vqa':
            ans_pred, ans_gt = answers[np.argmax(vqa_scores)], answers[label]
            json.dump({'question': question, 'ans_pred': ans_pred,
                       'ans_gt': ans_gt}, f)
        elif vis_type == 'loc':
            json.dump({'question': question, 'bbox_pred': list(bbox_pred),
                       'bbox_gt': list(bbox_gt)}, f)
        else:
            raise ValueError('Unknow vis_type ' + str(vis_type))
    print('visualization saved to ' + save_path)
    plt.close(h)


def vis_batch_vqa(model, data_reader, batch, vis_outputs, start_idx,
                  start_idx_correct, start_idx_incorrect, vis_dir):
    module_names = model.nmn.module_names
    answers = data_reader.batch_loader.answer_dict.word_list
    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
        num_correct = max(cfg.TEST.NUM_VIS_CORRECT-start_idx_correct, 0)
        num_incorrect = max(cfg.TEST.NUM_VIS_INCORRECT-start_idx_incorrect, 0)

        labels = batch['answer_label_batch']
        predictions = np.argmax(vis_outputs['vqa_scores'], axis=1)
        is_correct = predictions == labels
        inds = (list(np.where(is_correct)[0][:num_correct]) +
                list(np.where(~is_correct)[0][:num_incorrect]))
    else:
        num = min(len(batch['image_path_list']), cfg.TEST.NUM_VIS - start_idx)
        inds = range(num)
    for n in inds:
        img_path = batch['image_path_list'][n]
        if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
            if is_correct[n]:
                save_name = 'correct_%08d_%s.png' % (
                    start_idx_correct,
                    os.path.basename(img_path).split('.')[0])
                start_idx_correct += 1
            else:
                save_name = 'incorrect_%08d_%s.png' % (
                    start_idx_incorrect,
                    os.path.basename(img_path).split('.')[0])
                start_idx_incorrect += 1
        else:
            save_name = '%08d_%s.png' % (
                start_idx, os.path.basename(img_path).split('.')[0])
            start_idx += 1
        save_path = os.path.join(vis_dir, save_name)
        words = [
            data_reader.batch_loader.vocab_dict.idx2word(n_w) for n_w in
            batch['input_seq_batch'][:batch['seq_length_batch'][n], n]]
        vqa_scores = vis_outputs['vqa_scores'][n]
        label = batch['answer_label_batch'][n]
        txt_att = vis_outputs['txt_att'][n]
        att_stack = vis_outputs['att_stack'][n]
        stack_ptr = vis_outputs['stack_ptr'][n]
        module_prob = vis_outputs['module_prob'][n]
        if cfg.TEST.STEPWISE_VIS:
            vis_one_stepwise(img_path, words, module_names, txt_att, att_stack,
                             stack_ptr, module_prob, save_path, vis_type='vqa',
                             vqa_scores=vqa_scores, label=label,
                             answers=answers)
        else:
            vis_one_vqa(img_path, words, vqa_scores, label, module_names,
                        answers, txt_att, att_stack, stack_ptr, module_prob,
                        save_path)


def vis_batch_loc(model, data_reader, batch, vis_outputs, start_idx,
                  start_idx_correct, start_idx_incorrect, vis_dir):
    module_names = model.nmn.module_names
    iou_th = cfg.TEST.BBOX_IOU_THRESH
    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
        num_correct = max(cfg.TEST.NUM_VIS_CORRECT-start_idx_correct, 0)
        num_incorrect = max(cfg.TEST.NUM_VIS_INCORRECT-start_idx_incorrect, 0)

        bbox_pred = boxes.batch_feat_grid2bbox(
            np.argmax(vis_outputs['loc_scores'], axis=1),
            vis_outputs['bbox_offset'],
            data_reader.batch_loader.stride_H,
            data_reader.batch_loader.stride_W,
            data_reader.batch_loader.feat_H, data_reader.batch_loader.feat_W)
        bbox_gt = batch['bbox_batch']
        is_correct = boxes.batch_bbox_iou(bbox_pred, bbox_gt) >= iou_th
        inds = (list(np.where(is_correct)[0][:num_correct]) +
                list(np.where(~is_correct)[0][:num_incorrect]))
    else:
        num = min(len(batch['image_path_list']), cfg.TEST.NUM_VIS - start_idx)
        inds = range(num)
    for n in inds:
        img_path = batch['image_path_list'][n]
        if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
            if is_correct[n]:
                save_name = 'correct_%08d_%s.png' % (
                    start_idx_correct,
                    os.path.basename(img_path).split('.')[0])
                start_idx_correct += 1
            else:
                save_name = 'incorrect_%08d_%s.png' % (
                    start_idx_incorrect,
                    os.path.basename(img_path).split('.')[0])
                start_idx_incorrect += 1
        else:
            save_name = '%08d_%s.png' % (
                start_idx, os.path.basename(img_path).split('.')[0])
            start_idx += 1
        save_path = os.path.join(vis_dir, save_name)
        words = [
            data_reader.batch_loader.vocab_dict.idx2word(n_w) for n_w in
            batch['input_seq_batch'][:batch['seq_length_batch'][n], n]]
        loc_scores = vis_outputs['loc_scores'][n]
        bbox_offset = vis_outputs['bbox_offset'][n]
        bbox_pred = boxes.feat_grid2bbox(
            np.argmax(loc_scores), bbox_offset,
            data_reader.batch_loader.stride_H,
            data_reader.batch_loader.stride_W, data_reader.batch_loader.feat_H,
            data_reader.batch_loader.feat_W)
        bbox_gt = boxes.feat_grid2bbox(
            batch['bbox_ind_batch'][n], batch['bbox_offset_batch'][n],
            data_reader.batch_loader.stride_H,
            data_reader.batch_loader.stride_W, data_reader.batch_loader.feat_H,
            data_reader.batch_loader.feat_W)
        # bbox_gt = batch['bbox_batch'][n]
        txt_att = vis_outputs['txt_att'][n]
        att_stack = vis_outputs['att_stack'][n]
        stack_ptr = vis_outputs['stack_ptr'][n]
        module_prob = vis_outputs['module_prob'][n]
        if cfg.TEST.STEPWISE_VIS:
            vis_one_stepwise(img_path, words, module_names, txt_att, att_stack,
                             stack_ptr, module_prob, save_path, vis_type='loc',
                             loc_scores=loc_scores, bbox_pred=bbox_pred,
                             bbox_gt=bbox_gt)
        else:
            vis_one_loc(
                img_path, words, loc_scores, bbox_pred, bbox_gt, module_names,
                txt_att, att_stack, stack_ptr, module_prob, save_path)


def _print_bbox(bbox, color='r', scale_x=1., scale_y=1.):
    x1, y1, h, w = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color)


def _att_softmax(att):
    exps = np.exp(att - np.max(att))
    softmax = exps / np.sum(exps)
    return softmax


def attention_interpolation(im, att):
    softmax = _att_softmax(att)
    att_reshaped = skimage.transform.resize(softmax, im.shape[:2], order=3)
    # normalize the attention
    # make sure the 255 alpha channel is at least 3x uniform attention
    att_reshaped /= np.maximum(np.max(att_reshaped), 3. / att.size)
    att_reshaped = att_reshaped[..., np.newaxis]

    # make the attention area brighter than the rest of the area
    vis_im = att_reshaped * im + (1-att_reshaped) * im * .45
    vis_im = vis_im.astype(im.dtype)
    return vis_im


def _move_ptr_bw(stack_ptr):
    new_stack_ptr = np.zeros_like(stack_ptr)
    new_stack_ptr[:-1] = stack_ptr[1:]
    if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
        stack_bottom_mask = np.zeros_like(stack_ptr)
        stack_bottom_mask[0] = 1.
        new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def _read_two_from_stack(att_stack, stack_ptr):
    att_2 = np.sum(att_stack * stack_ptr, axis=-1)
    att_1 = np.sum(att_stack * _move_ptr_bw(stack_ptr), axis=-1)
    return att_1, att_2
