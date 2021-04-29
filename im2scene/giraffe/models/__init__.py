import torch.nn as nn
from im2scene.giraffe.models import (
    decoder, generator, bounding_box_generator, neural_renderer)


# Dictionaries
decoder_dict = {
    'simple': decoder.Decoder,
}

generator_dict = {
    'simple': generator.Generator,
}

background_generator_dict = {
    'simple': decoder.Decoder,
}

bounding_box_generator_dict = {
    'simple': bounding_box_generator.BoundingBoxGenerator,
}

neural_renderer_dict = {
    'simple': neural_renderer.NeuralRenderer
}


class GIRAFFE(nn.Module):
    ''' GIRAFFE model class.

    Args:
        device (device): torch device
        discriminator (nn.Module): discriminator network
        generator (nn.Module): generator network
        generator_test (nn.Module): generator_test network
    '''

    def __init__(self, device=None,
                 discriminator=None, generator=None, generator_test=None,
                 **kwargs):
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

    def forward(self, batch_size, **kwargs):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen(batch_size=batch_size)

    def generate_test_images(self):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen()

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
