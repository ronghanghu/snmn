import argparse
import os
import sys; sys.path.append('../../')  # NOQA
from glob import glob
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

from util.nets import resnet_v1, channel_mean


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

resnet101_model = '../tfmodel/resnet/resnet_v1_101.tfmodel'
image_basedir = '../clevr_loc_dataset/images/'
save_basedir = './resnet101_c4/'
H = 224
W = 224

image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
resnet101_c4 = resnet_v1.resnet_v1_101_c4(image_batch, is_training=False)
saver = tf.train.Saver()
sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver.restore(sess, resnet101_model)


def extract_image_resnet101_c4(impath):
    im = skimage.io.imread(impath)[..., :3]
    assert im.dtype == np.uint8
    im = skimage.transform.resize(im, [H, W], preserve_range=True)
    im_val = (im[np.newaxis, ...] - channel_mean)
    resnet101_c4_val = resnet101_c4.eval({image_batch: im_val}, sess)
    return resnet101_c4_val


def extract_dataset_resnet101_c4(image_dir, save_dir, ext_filter='*.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in enumerate(image_list):
        if (n_im+1) % 100 == 0:
            print('processing %d / %d' % (n_im+1, len(image_list)))
        image_name = os.path.basename(impath).split('.')[0]
        save_path = os.path.join(save_dir, image_name + '.npy')
        if not os.path.exists(save_path):
            resnet101_c4_val = extract_image_resnet101_c4(impath)
            np.save(save_path, resnet101_c4_val)


for image_set in ['loc_train', 'loc_val', 'loc_test']:
    print('Extracting image set ' + image_set)
    extract_dataset_resnet101_c4(
        os.path.join(image_basedir, image_set),
        os.path.join(save_basedir, image_set))
    print('Done.')
