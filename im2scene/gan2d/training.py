from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from torchvision.utils import save_image, make_grid
from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for the 2D-GAN.

    Args:
        model (nn.Module): 2D-GAN model
        optimizer (optimizer): generator optimizer
        optimizer_d (optimizer): discriminator optimizer
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): FID GT dictionary
        n_eval_iterations (int): number of evaluation iterations
    '''

    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 generator=None,
                 multi_gpu=False,  fid_dict={},
                 n_eval_iterations=10, **kwargs):
        self.model = model
        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir

        self.overwrite_visualization = True
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.visualize_z = torch.randn(
            16, self.generator.z_dim).to(device)

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''

        loss_g = self.train_step_generator(data, it)
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)
        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d,
            'd_real': real_d,
            'd_fake': fake_d,
        }

    def eval_step(self, data=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen(None).cpu())
        x_fake = torch.cat(x_fake, dim=0)
        x_fake = x_fake * 0.5 + 0.5
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)

        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        z = generator.sample_z()
        x_fake = generator(z)
        d_fake = discriminator(x_fake)
        gloss = compute_bce(d_fake, 1)
        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gloss.item()

    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            x_fake = generator(z)

        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (d_loss.item(), reg.item(), d_loss_fake.item(),
                d_loss_real.item())

    def visualize(self, it=0, **kwargs):
        ''' Visualize the data.

        '''
        self.model.generator.eval()
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator

        with torch.no_grad():
            image_fake = self.generator(self.visualize_z).cpu()
            # rescale
            image_fake = image_fake * 0.5 + 0.5

        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it

        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid
