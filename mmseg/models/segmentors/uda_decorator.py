# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Add img_interval
# - Add upscale_pred flag
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import torch
from copy import deepcopy
from typing import Optional, Union
from typing import Optional, Union
import random

from .teacher_module import EMATeacher
from .encoder_decoder import EncoderDecoder
from ..builder import build_segmentor

from torch import Tensor

from mmseg.models.utils.masking_transforms import build_mask_generator
from mmseg.registry import MODELS
from mmseg.utils import (SampleList, add_prefix)

OptSampleList = Optional[SampleList]

from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    return module


@MODELS.register_module()
class UDADecorator(EncoderDecoder):

    def __init__(self, **cfg):
        super(EncoderDecoder, self).__init__()

        self.cfg = cfg
        self.model = build_segmentor(deepcopy(cfg['model']))
        self.model.classes_to_concepts = cfg['vocalign']['classes_to_concepts']
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.use_thresh = cfg['model']['decode_head']['loss_decode'][
            'use_thresh']
        self.moving_threshold = torch.zeros(self.num_classes)
        self.moving_prior = torch.zeros(self.num_classes)
        self.gamma_chi = cfg['gamma_chi']
        self.zeta = cfg['zeta']
        self.beta = cfg['beta']

        self.use_masking = cfg['vocalign']['masking']
        self.random_masking = cfg['vocalign']['random']
        self.linear_masking = cfg['vocalign']['linear']
        self.mask_pseudo_threshold = cfg['vocalign']['mask_pseudo_threshold']
        self.mask_lambda = cfg['vocalign']['mask_lambda']
        assert self.linear_masking == False or self.random_masking == False, "Only one between random masking and linearly decreasing masking can be True"
        if not self.random_masking and not self.linear_masking:
            self.mask_gen = build_mask_generator(cfg['vocalign']['mask_generator'])

        self.class_guidance = cfg['vocalign']['class_guidance']
        self.classes_to_concepts = cfg['vocalign']['classes_to_concepts']
        self.topk_indices = None
        self.use_sample_prompt = cfg['vocalign']['use_sample_prompt']

        self.latent = cfg['vocalign']['latent']
        self.latent_lambda = cfg['vocalign']['latent_lambda']

        self.teacher = EMATeacher(
            use_mask_params=True,
            cfg=cfg)

        self.n_views = cfg['vocalign']['n_views']
        self.local_iter = 0

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas, upscale_pred)

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def train_step(self, data_batch, optim_wrapper, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        self.update_weights(model=self.model, iter=self.local_iter)
        self.batch_size = len(data_batch['inputs'])
        with optim_wrapper.optim_context(self):
            data = self.get_model().data_preprocessor(data_batch, True)
            losses = self._run_forward(data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def loss(self,
             inputs: Tensor,
             data_samples: SampleList,
             seg_weight=None) -> dict:
        H, W = inputs.shape[-2:]
        
        with torch.no_grad():
            masked_img = inputs
            if self.use_masking:
                if self.random_masking:
                    self.cfg['vocalign']['mask_generator']['mask_ratio'] = random.uniform(0, 0.7)
                    self.mask_gen = build_mask_generator(self.cfg['vocalign']['mask_generator'])
                elif self.linear_masking:
                    self.cfg['vocalign']['mask_generator']['mask_ratio'] = -0.00001*self.local_iter + 0.8
                    self.mask_gen = build_mask_generator(self.cfg['vocalign']['mask_generator'])

                masked_img = self.mask_gen.mask_image(masked_img)

            if self.latent:
                masked_lbl, masked_seg_weight, masked_prob, cost_lbl, topk, topk_templates = self.teacher(
                    inputs[:self.batch_size], data_samples, valid_pseudo_mask=None, latent=self.latent)
            else:
                masked_lbl, masked_seg_weight, masked_prob, topk, topk_templates = self.teacher(
                    inputs[:self.batch_size], data_samples, valid_pseudo_mask=None)
            for idx, sample in enumerate(data_samples):
                if idx < self.batch_size:
                    sample.masked_gt = masked_lbl[idx].unsqueeze(0)
                    if self.latent:
                        sample.latent_gt = cost_lbl[idx].unsqueeze(0)                      
                else:
                    sample.masked_gt = masked_lbl[(idx - self.batch_size) //
                                                self.n_views].unsqueeze(0)
                    if self.latent:
                        sample.latent_gt = cost_lbl[(idx - self.batch_size) // self.n_views].unsqueeze(0)

            if self.use_thresh:
                for lbl in torch.unique(masked_lbl):
                    lbl_indices = torch.where(masked_lbl == lbl)
                    m_star = masked_prob[lbl_indices].max()
                    chi_n = masked_prob[lbl_indices].shape[-1] / (H * W)
                    self.moving_prior[lbl] = self.gamma_chi * self.moving_prior[
                        lbl] + (1 - self.gamma_chi) * chi_n
                    self.moving_threshold[lbl] = self.zeta * (1 - torch.exp(
                        -self.moving_prior[lbl] / self.beta)) * m_star
                    
        masked_loss = self.get_model().loss(
            inputs[:self.batch_size],
            data_samples[self.batch_size:],
            masked_img[self.batch_size:],
            seg_weight=masked_prob,
            threshold=self.moving_threshold,
            batch_size=self.batch_size,
            n_views=self.n_views,
            latent=self.latent,
            topk=topk,
            topk_templates=topk_templates,
            sample_prompt=self.use_sample_prompt
        )

        if self.mask_lambda != 1:
            masked_loss['decode.loss_ce'] *= self.mask_lambda
        
        if 'latent.loss_ce' in masked_loss:
            masked_loss['latent.loss_ce'] *= self.latent_lambda

        self.local_iter += 1
        return masked_loss

    @torch.inference_mode()
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.get_model().data_preprocessor(data, False)
        return self.get_model()._run_forward(
            data, mode='predict')  # type: ignore

    @torch.inference_mode()
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.get_model().data_preprocessor(data, False)
        return self.get_model()._run_forward(
            data, mode='predict')  # type: ignore

@MODELS.register_module()
class TENTDecorator(EncoderDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        self.batch_size = len(data['inputs'])
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            data['inputs'] = data['inputs'][:self.batch_size]
            #data['data_samples'] = data['data_samples'][:self.batch_size]
            # Data shape mistmatch this way, hacky way of fixing. We don't use labels anyway!
            masked_lbl = data['inputs']
            for idx, sample in enumerate(data['data_samples']):
                if idx < self.batch_size:
                    sample.masked_gt = masked_lbl[idx]               
                else:
                    sample.masked_gt = masked_lbl[(idx - self.batch_size) // 1]
                    
            data['data_samples'] = data['data_samples'][self.batch_size:]

            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars