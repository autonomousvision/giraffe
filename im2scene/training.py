from collections import defaultdict
from torch import autograd
import torch.nn.functional as F
import numpy as np


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, *args, **kwargs):
        ''' Performs an evaluation.
        '''
        eval_list = defaultdict(list)

        # for data in tqdm(val_loader):
        eval_step_dict = self.eval_step()

        for k, v in eval_step_dict.items():
            eval_list[k].append(v)
        # eval_dict = {k: v for k, v in eval_list.items()}
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss
