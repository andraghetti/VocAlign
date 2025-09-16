# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import tempfile
from typing import List
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from ..utils import clip_wrapper, clip_model
from ..utils.clip_templates import (IMAGENET_TEMPLATES,
                                    IMAGENET_TEMPLATES_SELECT,
                                    IMAGENET_TEMPLATES_SELECT_CLIP)

from .lora import LoRA

from ..utils.tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


import os
import os.path as osp
from copy import deepcopy
import numpy as np
import time
import logging
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# TPT imports
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from mmengine.config import Config, ConfigDict
from mmengine.registry import RUNNERS, DATASETS, DATA_SAMPLERS, FUNCTIONS
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.logging import print_log
from mmengine.evaluator import Evaluator
from mmengine.dist import get_world_size, get_rank
from mmengine.model import is_model_wrapper
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.runner import Runner
from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.runner.utils import _get_batch_size
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.visualization import Visualizer
from mmengine.hooks import Hook

ConfigType = Union[Dict, Config, ConfigDict]

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

@MODELS.register_module()
class CLIPOVCATSeg(BaseModule):
    """CLIP based Open Vocabulary CAT-Seg model backbone.

    This backbone is the modified implementation of `CAT-Seg Backbone
    <https://arxiv.org/abs/2303.11797>`_. It combines the CLIP model and
    another feature extractor, a.k.a the appearance guidance extractor
    in the original `CAT-Seg`.

    Args:
        feature_extractor (ConfigType): Appearance guidance extractor
            config dict.
        train_class_json (str): The training class json file.
        test_class_json (str): The path to test class json file.
        clip_pretrained (str): The pre-trained clip type.
        clip_finetune (str): The finetuning settings of clip model.
        custom_clip_weights (str): The custmized clip weights directory. When
            encountering huggingface model download errors, you can manually
            download the pretrained weights.
        backbone_multiplier (float): The learning rate multiplier.
            Default: 0.01.
        prompt_depth (int): The prompt depth. Default: 0.
        prompt_length (int): The prompt length. Default: 0.
        prompt_ensemble_type (str): The prompt ensemble type.
            Default: "imagenet".
        pixel_mean (List[float]): The pixel mean for feature extractor.
        pxiel_std (List[float]): The pixel std for feature extractor.
        clip_pixel_mean (List[float]): The pixel mean for clip model.
        clip_pxiel_std (List[float]): The pixel std for clip model.
        clip_img_feat_size: (List[int]: Clip image embedding size from
            image encoder.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
            self,
            feature_extractor: ConfigType,
            train_class_json: str,
            test_class_json: str,
            clip_pretrained: str,
            clip_finetune: str,
            custom_clip_weights: str = None,
            backbone_multiplier=0.01,
            prompt_depth: int = 0,
            prompt_length: int = 0,
            prompt_ensemble_type: str = 'imagenet_select_clip',#'imagenet',#'imagenet_select_clip',
            pixel_mean: List[float] = [123.675, 116.280, 103.530],
            pixel_std: List[float] = [58.395, 57.120, 57.375],
            clip_pixel_mean: List[float] = [
                122.7709383, 116.7460125, 104.09373615
            ],
            clip_pixel_std: List[float] = [68.5005327, 66.6321579, 70.3231630],
            clip_img_feat_size: List[int] = [24, 24],
            init_cfg=None,
            design_details=None):
        super().__init__(init_cfg=init_cfg)
        # normalization parameters
        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(1, -1, 1, 1), False)
        self.register_buffer('clip_pixel_mean',
                             torch.Tensor(clip_pixel_mean).view(1, -1, 1, 1),
                             False)
        self.register_buffer('clip_pixel_std',
                             torch.Tensor(clip_pixel_std).view(1, -1, 1, 1),
                             False)
        self.clip_resolution = (
            384, 384) if clip_pretrained == 'ViT-B/16' else (336, 336)
        # modified clip image encoder with fixed size dense output
        self.clip_img_feat_size = clip_img_feat_size

        # prepare clip templates
        self.prompt_ensemble_type = prompt_ensemble_type
        if self.prompt_ensemble_type == 'imagenet_select':
            prompt_templates = IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == 'imagenet_select_clip':
            prompt_templates = IMAGENET_TEMPLATES_SELECT_CLIP
        elif self.prompt_ensemble_type == 'imagenet':
            prompt_templates = IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == 'single':
            prompt_templates = [
                'A photo of a {} in the scene',
            ]
        else:
            raise NotImplementedError
        self.prompt_templates = prompt_templates

        # build the feature extractor
        self.feature_extractor = MODELS.build(feature_extractor)

        # build CLIP model
        with open(train_class_json) as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json) as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts is not None
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        if clip_pretrained == 'ViT-G' or clip_pretrained == 'ViT-H':
            # for OpenCLIP models
            import open_clip
            name, pretrain = (
                'ViT-H-14',
                'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else (
                    'ViT-bigG-14', 'laion2b_s39b_b160k')
            try:
                open_clip_model = open_clip.create_model_and_transforms(
                    name,
                    pretrained=pretrain,
                    device=self.device,
                    force_image_size=336,
                )
                clip_model, _, clip_preprocess = open_clip_model
            except ConnectionError or LocalEntryNotFoundError as e:
                print(f'Has {e} when loading weights from huggingface!')
                print(
                    f'Will load {pretrain} weights from {custom_clip_weights}.'
                )
                assert custom_clip_weights is not None, 'Please specify custom weights directory.'  # noqa
                assert os.path.exists(
                    os.path.join(custom_clip_weights,
                                 'open_clip_pytorch_model.bin')
                ), 'Please provide a valid directory for manually downloaded model.'  # noqa
                open_clip_model = open_clip.create_model_and_transforms(
                    name,
                    pretrained=None,
                    device='cpu',
                    force_image_size=336,
                )
                clip_model, _, clip_preprocess = open_clip_model

                open_clip.load_checkpoint(
                    clip_model,
                    os.path.expanduser(
                        os.path.join(custom_clip_weights,
                                     'open_clip_pytorch_model.bin')))
                clip_model.to(torch.device(self.device))

            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip_wrapper.load(
                clip_pretrained,
                device=self.device,
                jit=False,
                prompt_depth=prompt_depth,
                prompt_length=prompt_length,
                design_details=design_details)

        # prepare CLIP model finetune
        self.clip_finetune = clip_finetune
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        self.text_features = None
        self.recompute = True

        for name, params in self.clip_model.named_parameters():
            print_log(name)
            if 'visual' in name:
                if clip_finetune == 'prompt':
                    params.requires_grad = True if 'prompt' in name else False
                elif clip_finetune == 'attention':
                    if 'attn' in name or 'position' in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == 'full':
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        finetune_backbone = backbone_multiplier > 0.
        for name, params in self.feature_extractor.named_parameters():
            if 'norm0' in name:
                params.requires_grad = False
            else:
                params.requires_grad = finetune_backbone

    def class_embeddings(self,
                         classnames,
                         templates,
                         clip_model,
                         device='cpu'):
        """Convert class names to text embeddings by clip model.

        Args:
            classnames (list): loaded from json file.
            templates (dict): text template.
            clip_model (nn.Module): prepared clip model.
            device (str | torch.device): loading device of text
                encoder results.
        """
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname)
                         for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(device)
            else:
                texts = clip_wrapper.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def custom_normalize(self, inputs, masked_inputs=None):
        """Input normalization for clip model and feature extractor
        respectively.

        Args:
            inputs: batched input images.
        """
        # clip images
        if not isinstance(masked_inputs, torch.Tensor):
            batched_clip = (inputs - self.clip_pixel_mean) / self.clip_pixel_std
        else:
            batched_clip = (masked_inputs - self.clip_pixel_mean) / self.clip_pixel_std
        batched_clip = F.interpolate(
            batched_clip,
            size=self.clip_resolution,
            mode='bilinear',
            align_corners=False)
        # feature extractor images
        batched = (inputs - self.pixel_mean) / self.pixel_std
        return batched, batched_clip

    def forward(self, inputs, masked_inputs=None, topk_indices=None, topk_templates=None, evaluate=False, sample_prompt=False):
        """
        Args:
            inputs: minibatch image. (B, 3, H, W)
        Returns:
            outputs (dict):
            'appearance_feat': list[torch.Tensor], w.r.t. out_indices of
                `self.feature_extractor`.
            'clip_text_feat': the text feature extracted by clip text encoder.
            'clip_text_feat_test': the text feature extracted by clip text
                encoder for testing.
            'clip_img_feat': the image feature extracted clip image encoder.
        """
        inputs, clip_inputs = self.custom_normalize(inputs, masked_inputs)
        outputs = dict()
        # extract appearance guidance feature
        outputs['appearance_feat'] = self.feature_extractor(inputs)

        #if (self.text_features is None or (not evaluate and not self.recompute)) and sample_prompt:
        #    with torch.no_grad():
        #        self.text_features = self.class_embeddings(self.class_texts,
        #                                            self.prompt_templates, self.clip_model,
        #                                            self.device).permute(1, 0, 2).float()
        if len(self.prompt_templates) != 80 and topk_templates is not None:
            raise NotImplementedError
        
        #temp value for randomk
        randomk = 40
        
        if evaluate:
            if self.recompute:
                self.text_features = self.class_embeddings(self.test_class_texts,
                                                        self.prompt_templates, self.clip_model,
                                                        self.device).permute(1, 0, 2).float()
            text_features = self.text_features
            self.recompute = False
        else:
            text_features = None
            self.recompute = True
            # If not sample prompte use all prompt templates
            if not sample_prompt:
                templates = self.prompt_templates

            # Use the TopK indices from the teacher module to pick which classes to use
            if topk_indices is not None:
                for i, indices in enumerate(topk_indices):
                    class_texts = []
                    for index in indices:
                        class_texts.append(self.class_texts[index])

                    # If sample prompt produce all prompt templates for relevant classes
                    if sample_prompt:
                     # Compute most of the embeddings with no grad
                        with torch.no_grad():
                            self.text_features = self.class_embeddings(class_texts,
                                                                    self.prompt_templates, self.clip_model,
                                                                    self.device).permute(1, 0, 2).float()
                        # Use topk or random templates
                        if topk_templates is None:
                            template_indices, _ = torch.sort(torch.randperm(len(self.prompt_templates))[:randomk])
                            templates = [self.prompt_templates[index] for index in template_indices]
                        else:
                            template_indices = topk_templates[i]
                            templates = [self.prompt_templates[index] for index in template_indices]

                    features = self.class_embeddings(class_texts,
                                                    templates, self.clip_model,
                                                    self.device).permute(1, 0, 2).float()

                    # Prompt sampling for reduced memory usage
                    if sample_prompt:
                        self.text_features[:, template_indices] = features
                        features = self.text_features

                    if text_features is None:
                        text_features = features.unsqueeze(0)
                    else:
                        text_features = torch.cat((text_features, features.unsqueeze(0)))
            else:
                # If sample prompt produce all prompt templates for relevant classes
                if sample_prompt:
                    with torch.no_grad():
                        self.text_features = self.class_embeddings(self.class_texts,
                                                                self.prompt_templates, self.clip_model,
                                                                self.device).permute(1, 0, 2).float()
                    for i in range(inputs.shape[0]):
                        # Use topk or random templates
                        if topk_templates is None:
                            template_indices, _ = torch.sort(torch.randperm(len(self.prompt_templates))[:randomk])
                            templates = [self.prompt_templates[index] for index in template_indices]
                        else:
                            template_indices = topk_templates[i]
                            templates = [self.prompt_templates[index] for index in template_indices]

                        features = self.class_embeddings(self.class_texts,
                                                                templates, self.clip_model,
                                                                self.device).permute(1, 0, 2).float()
                        # Replace the sampled prompts with the the new features and add the non sampled previous prompts to the features
                        self.text_features[:, template_indices] = features
                        features = self.text_features
                        if text_features is None:
                            text_features = features.unsqueeze(0)
                        else:
                            text_features = torch.cat((text_features, features.unsqueeze(0)))
                else:
                    # Otherwise compute text features regularly
                    text_features = self.class_embeddings(self.class_texts,
                                                                self.prompt_templates, self.clip_model,
                                                                self.device).permute(1, 0, 2).float()
                
        self.clip_model.visual.transformer.lora_w = self.clip_model.transformer.lora_w
        # extract clip features
        outputs['clip_text_feat'] = text_features
        outputs['clip_text_feat_test'] = text_features
        clip_features = self.clip_model.encode_image(
            clip_inputs, dense=True)  # B, 577(24x24+1), C
        B = clip_features.size(0)
        outputs['clip_img_feat'] = clip_features[:, 1:, :].permute(
            0, 2, 1).reshape(B, -1, *self.clip_img_feat_size)
        return outputs

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu(cfg):
    backbone_name = cfg.model.backbone.clip_pretrained
    url = clip_wrapper._MODELS[backbone_name]
    model_path = clip_wrapper._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip_model.build_model(state_dict or model.state_dict())

    return model

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.RandomHorizontalFlip(),
        ])
