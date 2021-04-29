#!/usr/bin/env python3
'''
Code is mainly adopted from :
https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
'''

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import sys
sys.path.append('.')
from im2scene.eval import calculate_activation_statistics
import numpy as np
import torch 
import cv2 
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import lmdb
import random
from torchvision.utils import save_image, make_grid

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--class-name', type=str, default='',
                    help='If only this class should be considered')
parser.add_argument('--image-folder', type=str, default='',
                    help='Name of image folder (empty string means no subfolder)')
parser.add_argument('--max-n-images', type=int, default=20000,
                    help='Maximal images to use')
parser.add_argument('--dims', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--npz-file', default=False, type=bool,
                    help='whether the images are saved in npz file')
parser.add_argument('--lsun', default=False, type=bool,
                    help='whether the images are saved in lsun style')
parser.add_argument('--img-size', default=64, type=int,
                    help='which resolution of images (only relevant for LSUN)')
parser.add_argument('--out-file', default=None, type=str,
                    help='specific folder where to save npz file')
parser.add_argument('--regex', default=False, type=bool,
                    help='specific if input path is already a regex')

def _compute_statistics_of_path(path, class_name, image_folder, batch_size, dims, cuda, max_n_images, npz_file, lsun, size, out_file_path, regex):

    if regex:
        classes = ['0']
    else:
        if len(class_name) > 0:
            classes = [class_name]
        else:
            classes = [cl for cl in os.listdir(
                path) if os.path.isdir(os.path.join(path, cl))]

    files = []
    for cl in classes:
        cl_path = os.path.join(path, cl)
        
        if npz_file:
            f = glob.glob(os.path.join(cl_path, '*.npz'))[0]
            imgs = np.load(f)['images'].astype(np.float32)[..., :3] / 255
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            files.append(imgs)
        elif lsun:
            if "church_outdoor_train_lmdb" in class_name:
                transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((size, size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                ])

            x = lmdb.open(cl_path, map_size=1099511627776, max_readers=100, readonly=True, lock=False)
            with x.begin(write=False) as env:
                cursor = env.cursor()
                # n_iter = env.stat()['entries']
                for idx, (k, v) in tqdm(enumerate(cursor), total=max_n_images):  
                    img = cv2.cvtColor(cv2.imdecode(np.fromstring(v, dtype=np.uint8), 1),  cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img).convert("RGB")
                    img = transform(img)
                    img = img.unsqueeze(0)
                    files.append(img)
                    if idx ==max_n_images:
                        break
        elif regex:
            # check for celebA
            if "celeba_hq" in path:
                transform = transforms.Compose([
                            transforms.CenterCrop(650),
                            transforms.Resize((size, size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                ])    
            elif "celebA" in path:
                transform = transforms.Compose([
                            transforms.CenterCrop(108),
                            transforms.Resize((size, size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                ])    
            elif "comprehensive_cars" in path:
                transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.RandomCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])         
            else:
                transform = transforms.Compose([
                            transforms.Resize((size, size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                ])
            f = glob.glob(path)
            random.shuffle(f)
            files = []
            data_type = os.path.basename(path).split(".")[-1]

            print("Loading files ...")
            for idx, fp in tqdm(enumerate(f)):
                if data_type == 'npy':
                    img = np.load(fp)[0].transpose(1, 2, 0)
                    img = Image.fromarray(img).convert("RGB")
                else:
                    img = Image.open(fp).convert("RGB")
                img = transform(img)
                img = img.unsqueeze(0)
                files.append(img)
                if idx == max_n_images-1:
                    break
        else:
            models = [m for m in os.listdir(
                cl_path) if os.path.isdir(os.path.join(cl_path, m))]
            for m in models:
                model_path = os.path.join(cl_path, m)
                if len(image_folder) > 0:
                    model_path = os.path.join(model_path, image_folder)

                files += glob.glob(os.path.join(model_path, '*.png'))
                files += glob.glob(os.path.join(model_path, '*.jpg'))

    if npz_file or lsun or regex:
        files = torch.cat(files, dim=0)
    len_files = len(files) if type(files) == list else files.shape[0]

    print('Found %d images.' % len_files)
    if len(files) > max_n_images:
        idx = np.random.choice(len(files), size=(max_n_images,), replace=False)
        if npz_file or lsun or regex:
            files = files[idx]
        else:
            files = [files[i] for i in idx]

    # Save evaluation images
    outfile_image = out_file_path[:-4] + '_images.npz'
    np.savez(outfile_image, images=files)

    # Save visualisation file
    outfile_vis = out_file_path[:-4] + '_vis.jpg'
    save_image(make_grid(files[:256], nrow=16), outfile_vis)
    print("Start Calculations with %d images..." % len(files))
    m, s = calculate_activation_statistics(files, batch_size,
                                           dims, cuda)
    print('finished!')

    out_dict = {'m': m, 's': s, }
    if out_file_path is not None and len(out_file_path) > 0:
        out_file = out_file_path
    elif len(class_name) > 0:
        out_file = os.path.join(path, class_name, 'fid_statistics.npz')
    else:
        out_file = os.path.join(path, 'fid_statistics.npz')

    np.savez(out_file, **out_dict)
    print("Saved dictionary to %s." % out_file)

    return m, s


def save_fid_to_file(path, class_name, image_folder, batch_size, cuda, dims, max_n_images, npz_file, lsun, img_size, out_file_path, regex):
    """Calculates the FID of two paths"""
    # for p in paths:
    #     if not os.path.exists(p):
    #         raise RuntimeError('Invalid path: %s' % p)

    m1, s1 = _compute_statistics_of_path(path, class_name, image_folder, batch_size,
                                         dims, cuda, max_n_images, npz_file, lsun, img_size, out_file_path, regex)
    # m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
    #                                      dims, cuda)
    # fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    # return fid_value


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert(args.out_file is not None)

    save_fid_to_file(args.path,
                     args.class_name,
                     args.image_folder,
                     args.batch_size,
                     args.gpu != '',
                     args.dims,
                     args.max_n_images,
                     args.npz_file,
                     args.lsun,
                     args.img_size,
                     args.out_file,
                     args.regex)
