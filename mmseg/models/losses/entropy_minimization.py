# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss

@MODELS.register_module()
class EntropyMinimization(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 pixels=False,
                 images=False,
                 align=False,
                 class_weight=None,
                 top_percentage=1.0,
                 loss_weight=1.0,
                 loss_name='entropy_minimization_loss'):
        super().__init__()
        assert (pixels is False) or (images is False)
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.pixels = pixels
        self.images = images
        self.align = align
        self.percentage = top_percentage

        if self.pixels:
            self.filtering = self.select_confident_pixels
            self.cls_criterion = self.avg_entropy_pixels
        elif self.images:
            self.filtering = self.select_confident_images
            self.cls_criterion = self.avg_entropy
        else:
            raise Exception("One between pixels and images must be set to True")
        self._loss_name = loss_name

    def select_confident_images(self, logits, topTPT):     
        batch_entropy = -(logits.flatten(2,3).softmax(1) * logits.flatten(2,3).log_softmax(1)).sum(1)
        avg_batch_entropy = torch.mean(batch_entropy, dim=1)

        idxTPT = torch.argsort(avg_batch_entropy,  descending=False)[:int(np.ceil(batch_entropy.size()[0] * topTPT))]
        # idxAlign = torch.argsort(avg_batch_entropy,  descending=False)[:int(np.ceil(batch_entropy.size()[0] * topAlign))]

        return logits[idxTPT, ...]
    
    def select_confident_pixels(self, logits, topTPT):
        batch_pixelwise_entropy = -(logits.flatten(2,3).softmax(1) * logits.flatten(2,3).log_softmax(1)).sum(1)
        avg_pixelwise_entropy = torch.mean(batch_pixelwise_entropy, dim=0)

        idxTPT = torch.argsort(avg_pixelwise_entropy,  descending=False)[:int(np.ceil(batch_pixelwise_entropy.size()[1] * topTPT))]
        # idxAlign = torch.argsort(avg_pixelwise_entropy,  descending=False)[:int(np.ceil(batch_pixelwise_entropy.size()[1] * topAlign))]

        return logits.flatten(2,3)[..., idxTPT]

    def avg_entropy(self, outputs, ignore_index=None, weight=None):
        logits = outputs - outputs.logsumexp(dim=1, keepdim=True) # operation along the class dimension, logits = outputs.log_softmax(dim=1) [N, 1000] 
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # operation along the batch dimension, avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        pixel_wise_loss = -(avg_logits * torch.exp(avg_logits)).flatten(-2, -1).sum(dim=0)
        return torch.mean(pixel_wise_loss)
    
    def avg_entropy_pixels(self, outputs, ignore_index=None, weight=None):
        logits = outputs - outputs.logsumexp(dim=1, keepdim=True) # operation along the class dimension, logits = outputs.log_softmax(dim=1) [N, 1000] 
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # operation along the batch dimension, avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        pixel_wise_loss = -(avg_logits * torch.exp(avg_logits)).sum(dim=0)
        return torch.mean(pixel_wise_loss)
    
    def distr_align_loss(self, out_feat, targ_feat, layers_from=0, layers_to=12, moments=5):
        '''
        A feature distibution alignment L1 loss between mean and variance of the features
        '''
        distr_loss = 0
        out_means, out_vars = out_feat
        targ_means, targ_vars = targ_feat
        transf_layers = layers_to
        for l in range(layers_from, transf_layers-1):
            out_mean, out_var = out_means[l], out_vars[l]
            targ_mean, targ_var = targ_means[l], targ_vars[l]
            distr_loss += 0.5 * F.l1_loss(out_mean, targ_mean) + 0.5 * F.l1_loss(out_var, targ_var)
        return distr_loss

    def forward(self,
                logits,
                batch_data_samples,
                **kwargs):
        """Forward function."""
        valid_image = self.filtering(logits=logits, topTPT=self.percentage)

        loss_cls = self.loss_weight * self.cls_criterion(
            valid_image,
            **kwargs)

        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    

@MODELS.register_module()
class EntropyMinimizationTENT(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 align=False,
                 class_weight=None,
                 top_percentage=1.0,
                 loss_weight=1.0,
                 loss_name='entropy_minimization_loss'):
        super().__init__()
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.align = align
        self.percentage = top_percentage
        self._loss_name = loss_name

    def forward(self,
                logits,
                batch_data_samples,
                **kwargs):
        """Forward function."""
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        loss_cls = self.loss_weight * torch.mean(loss)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name