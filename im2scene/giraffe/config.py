import os
from im2scene.discriminator import discriminator_dict
from im2scene.giraffe import models, training, rendering
from copy import deepcopy
import numpy as np


def get_model(cfg, device=None, len_dataset=0, **kwargs):
    ''' Returns the giraffe model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        len_dataset (int): length of dataset
    '''
    decoder = cfg['model']['decoder']
    discriminator = cfg['model']['discriminator']
    generator = cfg['model']['generator']
    background_generator = cfg['model']['background_generator']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    discriminator_kwargs = cfg['model']['discriminator_kwargs']
    generator_kwargs = cfg['model']['generator_kwargs']
    background_generator_kwargs = \
        cfg['model']['background_generator_kwargs']
    bounding_box_generator = cfg['model']['bounding_box_generator']
    bounding_box_generator_kwargs = \
        cfg['model']['bounding_box_generator_kwargs']
    neural_renderer = cfg['model']['neural_renderer']
    neural_renderer_kwargs = cfg['model']['neural_renderer_kwargs']
    z_dim = cfg['model']['z_dim']
    z_dim_bg = cfg['model']['z_dim_bg']
    img_size = cfg['data']['img_size']

    # Load always the decoder
    decoder = models.decoder_dict[decoder](
        z_dim=z_dim, **decoder_kwargs
    )

    if discriminator is not None:
        discriminator = discriminator_dict[discriminator](
            img_size=img_size, **discriminator_kwargs)
    if background_generator is not None:
        background_generator = \
            models.background_generator_dict[background_generator](
                z_dim=z_dim_bg, **background_generator_kwargs)
    if bounding_box_generator is not None:
        bounding_box_generator = \
            models.bounding_box_generator_dict[bounding_box_generator](
                z_dim=z_dim, **bounding_box_generator_kwargs)
    if neural_renderer is not None:
        neural_renderer = models.neural_renderer_dict[neural_renderer](
            z_dim=z_dim, img_size=img_size, **neural_renderer_kwargs
        )

    if generator is not None:
        generator = models.generator_dict[generator](
            device, z_dim=z_dim, z_dim_bg=z_dim_bg, decoder=decoder,
            background_generator=background_generator,
            bounding_box_generator=bounding_box_generator,
            neural_renderer=neural_renderer, **generator_kwargs)

    if cfg['test']['take_generator_average']:
        generator_test = deepcopy(generator)
    else:
        generator_test = None

    model = models.GIRAFFE(
        device=device,
        discriminator=discriminator, generator=generator,
        generator_test=generator_test,
    )
    return model


def get_trainer(model, optimizer, optimizer_d, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    overwrite_visualization = cfg['training']['overwrite_visualization']
    multi_gpu = cfg['training']['multi_gpu']
    n_eval_iterations = (
        cfg['training']['n_eval_images'] // cfg['training']['batch_size'])

    fid_file = cfg['data']['fid_file']
    assert(fid_file is not None)
    fid_dict = np.load(fid_file)

    trainer = training.Trainer(
        model, optimizer, optimizer_d, device=device, vis_dir=vis_dir,
        overwrite_visualization=overwrite_visualization, multi_gpu=multi_gpu,
        fid_dict=fid_dict,
        n_eval_iterations=n_eval_iterations,
    )

    return trainer


def get_renderer(model, cfg, device, **kwargs):
    ''' Returns the renderer object.

    Args:
        model (nn.Module): GIRAFFE model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    renderer = rendering.Renderer(
        model,
        device=device,)
    return renderer
