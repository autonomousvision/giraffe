import torch.nn as nn
from im2scene.gan2d.models import generator


generator_dict = {
    'simple': generator.Generator,
}


class GAN2D(nn.Module):
    ''' 2D-GAN model class.

    Args:
        device (device): torch device
        discriminator (nn.Module): discriminator network
        generator (nn.Module): generator network
        generator_test (nn.Module): generator_test network
    '''

    def __init__(self, device=None, discriminator=None, generator=None,
                 generator_test=None, **kwargs):
        super().__init__()

        if discriminator is not None:
            self.discriminator = discriminator.to(device)
        else:
            self.discriminator = None
        if generator is not None:
            self.generator = generator.to(device)
        else:
            self.generator = None

        if generator_test is not None:
            self.generator_test = generator_test.to(device)
        else:
            self.generator_test = None

    def forward(self, *args, **kwargs):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        images = gen(None)
        images = images * 0.5 + 0.5  # scale to [0, 1]
        return images

    def generate_test_images(self):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        images = gen(None)
        images = images * 0.5 + 0.5  # scale to [0, 1]
        return images

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
