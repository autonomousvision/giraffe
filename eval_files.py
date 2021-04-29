import torch
import os
import argparse
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(
    description='Evaluate your own generated images (see ReadMe for more\
                 information).'
)
parser.add_argument('--input-file', type=str, help='Path to input file.')
parser.add_argument('--gt-file', type=str, help='Path to gt file.')
parser.add_argument('--n-images', type=int, default=20000,
                    help='Number of images used for evaluation.')

args = parser.parse_args()
n_images = args.n_images


def load_np_file(np_file):
    ext = os.path.basename(np_file).split('.')[-1]
    assert(ext in ['npy'])
    if ext == 'npy':
        return torch.from_numpy(np.load(np_file)).float() / 255


img_fake = load_np_file(args.input_file)[:n_images]
fid_dict = np.load(args.gt_file)
out_dict_file = os.path.join(
    os.path.dirname(args.input_file), 'fid_evaluation.npz')
out_vis_file = os.path.join(
    os.path.dirname(args.input_file), 'fid_evaluation.jpg')
out_dict = {}

print("Start FID calculation with %d images ..." % img_fake.shape[0])
mu, sigma = calculate_activation_statistics(img_fake)
out_dict['m'] = mu
out_dict['sigma'] = sigma

fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
out_dict['fid'] = fid_score
print("FID Score (%d images): %.6f" % (n_images, fid_score))
np.savez(out_dict_file, **out_dict)

save_image(make_grid(img_fake[:256], nrow=16, pad_value=1.), out_vis_file)
