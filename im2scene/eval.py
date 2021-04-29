import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
from tqdm import tqdm
from im2scene.inception import InceptionV3

'''
NOTE: The code is largely adopted from:
https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/fid_score.py
'''


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """

    model.eval()

    is_list = type(files) == list

    if is_list:
        if batch_size > len(files):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(files)

        pred_arr = np.empty((len(files), dims))

        for i in tqdm(range(0, len(files), batch_size)):
            start = i
            end = i + batch_size

            images = np.array([imread(str(f)).astype(np.float32)
                               for f in files[start:end]])
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            images /= 255

            batch = torch.from_numpy(images).type(torch.FloatTensor)

            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu(
            ).data.numpy().reshape(pred.size(0), -1)
    else:
        # assume files is already tensor of size N x 3 x H x W and [0, 1]
        pred_arr = np.empty((files.shape[0], dims))
        start_idx = 0
        for batch in torch.split(files, batch_size, dim=0):
            if cuda:
                batch = batch.cuda()
            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
            # pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
            #pred_arr.append(pred.cpu().data.numpy().reshape(pred.shape[0], -1))
        #pred_arr = np.concatenate(pred_arr, axis=0)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, batch_size=50,
                                    dims=2048, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    if cuda:
        model = model.cuda()

    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
