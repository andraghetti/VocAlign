# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
from copy import deepcopy

import numpy as np
import torch
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd

from mmseg.models import build_segmentor


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    return module


class EMATeacher(Module):

    def __init__(self, use_mask_params, cfg):
        super(EMATeacher, self).__init__()
        prefix = 'mask_' if use_mask_params else ''
        self.alpha = cfg['vocalign'][f'{prefix}alpha']
        if self.alpha == 'same':
            self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['vocalign'][f'{prefix}pseudo_threshold']
        if self.pseudo_threshold == 'same':
            self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['vocalign']['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['vocalign']['pseudo_weight_ignore_bottom']
        self.class_guidance = cfg['vocalign']['class_guidance']
        self.classes_to_concepts = cfg['vocalign']['classes_to_concepts']
        self.concepts_json = cfg['vocalign']['concepts_json']
        self.topk = cfg['vocalign']['topk']
        self.randomk = cfg['vocalign']['randomk']

        ema_cfg = deepcopy(cfg['model'])

        if self.class_guidance:
            ema_cfg['backbone']['train_class_json'] = self.concepts_json
            ema_cfg['backbone']['test_class_json'] = self.concepts_json
            ema_cfg['decode_head']['num_classes'] = len(
                self.classes_to_concepts)
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self, model):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def get_pseudo_label_and_weight(self, logits):
        B, _, H, W = logits.shape
        ema_softmax = torch.softmax(logits.detach(), dim=1)

        # Create TopK labels + random labels
        topk_indices = None
        if self.topk is not None:
            prediction_mean = torch.mean(ema_softmax, dim=(2, 3))
            _, topk_indices = torch.topk(prediction_mean, k=self.topk, dim=1, largest=True) 
            
            # pick random indices
            if self.randomk is not None:
                new_indices = torch.zeros((B, self.topk + self.randomk), device=topk_indices.device, dtype=topk_indices.dtype)
                for i, indices in enumerate(topk_indices):
                    random_indices = torch.randperm(prediction_mean.shape[1])
                    random_indices = torch.tensor([x for x in random_indices if x not in indices], device=prediction_mean.device)[:self.randomk]
                    new_indices[i] = torch.cat((topk_indices[i], random_indices), dim=0)
                topk_indices = new_indices
            # sort
            topk_indices, _ = torch.sort(topk_indices)

            topk_gather = topk_indices.unsqueeze(-1).unsqueeze(-1)
            topk_gather = topk_gather.expand(-1, -1, ema_softmax.size(2), ema_softmax.size(3))
            ema_softmax = torch.gather(ema_softmax, dim=1, index=topk_gather)

        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)

        return pseudo_label, pseudo_weight, pseudo_prob, topk_indices

    def get_guided_pseudo_label_and_weight(self, logits):
        B, _, H, W = logits.shape
        agg_pred = torch.zeros(
            B, len(self.classes_to_concepts), H, W, device=logits.device)
        # Try doing softmax first
        softmax_logits = torch.softmax(logits, dim=1)
        for cls_i, conc_i in self.classes_to_concepts.items():
            agg_pred[:, cls_i] = softmax_logits[:, conc_i].sum(dim=1)

        # Create TopK labels + random labels
        topk_indices = None
        if self.topk is not None:
            prediction_mean = torch.mean(agg_pred, dim=(2, 3))
            _, topk_indices = torch.topk(prediction_mean, k=self.topk, dim=1, largest=True) 
            
            # pick random indices
            if self.randomk is not None:
                new_indices = torch.zeros((B, self.topk + self.randomk), device=topk_indices.device, dtype=topk_indices.dtype)
                for i, indices in enumerate(topk_indices):
                    random_indices = torch.randperm(prediction_mean.shape[1])
                    random_indices = torch.tensor([x for x in random_indices if x not in indices], device=prediction_mean.device)[:self.randomk]
                    new_indices[i] = torch.cat((topk_indices[i], random_indices), dim=0)
                topk_indices = new_indices
            # sort
            topk_indices, _ = torch.sort(topk_indices)

            topk_gather = topk_indices.unsqueeze(-1).unsqueeze(-1)
            topk_gather = topk_gather.expand(-1, -1, agg_pred.size(2), agg_pred.size(3))
            agg_pred = torch.gather(agg_pred, dim=1, index=topk_gather) 
        
        #ema_softmax = torch.softmax(agg_pred.detach(), dim=1)
        ema_softmax = agg_pred.detach()
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight, pseudo_prob, topk_indices
    
    def agg_cost_vol_concepts(self, cost_volume):
        B, _, _, H, W = cost_volume.shape
        agg_cost = torch.zeros(
            B, len(self.classes_to_concepts), H, W, device=cost_volume.device)
        cost_volume = cost_volume.sum(dim=1)
        for cls_i, conc_i in self.classes_to_concepts.items():
            agg_cost[:, cls_i] = cost_volume[:, conc_i].max(dim=1).values
        agg_cost = agg_cost.softmax(1).max(1)[1]
        return agg_cost

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if iter > 0:
            self._update_ema(model, iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

    def __call__(self, target_img, target_img_metas, valid_pseudo_mask, latent=False):
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
        
        # TOPK here
        if latent:
            ema_logits, cost_volume = self.get_ema_model().generate_pseudo_label(
                img=target_img, img_metas=target_img_metas, latent=latent)
            cost_lbl = self.agg_cost_vol_concepts(cost_volume=cost_volume)
        else:
            ema_logits, topk_templates = self.get_ema_model().generate_pseudo_label(
                img=target_img, img_metas=target_img_metas, latent=False)
        # and here
        if not self.class_guidance:
            pseudo_label, pseudo_weight, pseudo_prob, topk = self.get_pseudo_label_and_weight(
                ema_logits)
        else:
            pseudo_label, pseudo_weight, pseudo_prob, topk = self.get_guided_pseudo_label_and_weight(
                ema_logits)
        del ema_logits
        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)

        if latent:
            return pseudo_label, pseudo_weight, pseudo_prob, cost_lbl, topk
        else:
            return pseudo_label, pseudo_weight, pseudo_prob, topk, topk_templates
