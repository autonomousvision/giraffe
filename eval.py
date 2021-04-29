import torch
import os
import argparse
from tqdm import tqdm
import time
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import numpy as np
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from math import ceil
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(
    description='Evaluate a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
out_dict_file = os.path.join(out_dir, 'fid_evaluation.npz')
out_img_file = os.path.join(out_dir, 'fid_images.npy')
out_vis_file = os.path.join(out_dir, 'fid_images.jpg')

# Model
model = config.get_model(cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])


# Generate
model.eval()

fid_file = cfg['data']['fid_file']
assert(fid_file is not None)
fid_dict = np.load(cfg['data']['fid_file'])

n_images = cfg['test']['n_images']
batch_size = cfg['training']['batch_size']
n_iter = ceil(n_images / batch_size)

out_dict = {'n_images': n_images}

img_fake = []
t0 = time.time()
for i in tqdm(range(n_iter)):
    with torch.no_grad():
        img_fake.append(model(batch_size).cpu())
img_fake = torch.cat(img_fake, dim=0)[:n_images]
img_fake.clamp_(0., 1.)
n_images = img_fake.shape[0]

t = time.time() - t0
out_dict['time_full'] = t
out_dict['time_image'] = t / n_images

img_uint8 = (img_fake * 255).cpu().numpy().astype(np.uint8)
np.save(out_img_file[:n_images], img_uint8)

# use uint for eval to fairly compare
img_fake = torch.from_numpy(img_uint8).float() / 255.
mu, sigma = calculate_activation_statistics(img_fake)
out_dict['m'] = mu
out_dict['sigma'] = sigma

# calculate FID score and save it to a dictionary
fid_score = calculate_frechet_distance(mu, sigma, fid_dict['m'], fid_dict['s'])
out_dict['fid'] = fid_score
print("FID Score (%d images): %.6f" % (n_images, fid_score))
np.savez(out_dict_file, **out_dict)

# Save a grid of 16x16 images for visualization
save_image(make_grid(img_fake[:256], nrow=16, pad_value=1.), out_vis_file)
