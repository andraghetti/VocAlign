# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict, defaultdict
from typing import Tuple, Union, Optional
import math

import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_, zeros_
import torch.nn.functional as F


from .functional import multi_head_attention_forward as lora_multi_head_attention_forward


class Bottleneck(nn.Module):
    """Custom implementation of Bottleneck in ResNet."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # all conv layers have stride 1.
        # an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): the input feature.
        """
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """Attention Pool2d."""

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): the input feature.
        """
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)


class MultiheadAttentionLoRA(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, add_lora=False) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)

        self.added_lora = add_lora

    def _add_lora(self, device, shared, text_vis_proj=None, text_lora=None, r=1, alpha:Optional[int]=1, requires_grad=True):
        self.shared = shared
        self.rank = r
        self.alpha = alpha
        self.added_lora = True
     
        if not self.shared:
            self.a_proj_weight_lora_q = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_proj_weight_lora_q = Parameter(torch.empty((self.embed_dim, r)).to(device))
            self.a_proj_weight_lora_k = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_proj_weight_lora_k = Parameter(torch.empty((self.embed_dim, r)).to(device))
            self.a_proj_weight_lora_v = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_proj_weight_lora_v = Parameter(torch.empty((self.embed_dim, r)).to(device))
            
            self._reset_lora()

            if not requires_grad:
                self.a_proj_weight_lora_q.requires_grad = False
                self.b_proj_weight_lora_q.requires_grad = False
                self.a_proj_weight_lora_k.requires_grad = False
                self.b_proj_weight_lora_k.requires_grad = False
                self.a_proj_weight_lora_v.requires_grad = False
                self.b_proj_weight_lora_v.requires_grad = False
        else:
            self.a_init_lora_q = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_init_lora_q = Parameter(torch.empty((self.embed_dim, r)).to(device))
            self.a_init_lora_k = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_init_lora_k = Parameter(torch.empty((self.embed_dim, r)).to(device))
            self.a_init_lora_v = Parameter(torch.empty((r, self.embed_dim)).to(device))
            self.b_init_lora_v = Parameter(torch.empty((self.embed_dim, r)).to(device))

            self._reset_init_lora()

            self.text_vis_proj = text_vis_proj
            
            text_a_q = text_lora[0]
            text_b_q = text_lora[1]
            text_a_k = text_lora[2]
            text_b_k = text_lora[3]
            text_a_v = text_lora[4]
            text_b_v = text_lora[5]
            
            residuals_lora_q = self.text_vis_proj(torch.concatenate((text_a_q, text_b_q.T)).reshape(self.rank, -1))
            residuals_lora_k = self.text_vis_proj(torch.concatenate((text_a_k, text_b_k.T)).reshape(self.rank, -1))
            residuals_lora_v = self.text_vis_proj(torch.concatenate((text_a_v, text_b_v.T)).reshape(self.rank, -1))

            if not requires_grad:
                self.a_init_lora_q.requires_grad = False
                self.b_init_lora_q.requires_grad = False
                self.a_init_lora_k.requires_grad = False
                self.b_init_lora_k.requires_grad = False
                self.a_init_lora_v.requires_grad = False
                self.b_init_lora_v.requires_grad = False

            self.a_proj_weight_lora_q = self.a_init_lora_q + residuals_lora_q[:, :self.embed_dim]
            self.b_proj_weight_lora_q = self.b_init_lora_q + residuals_lora_q[:, self.embed_dim:].T
            self.a_proj_weight_lora_k = self.a_init_lora_k + residuals_lora_k[:, :self.embed_dim]
            self.b_proj_weight_lora_k = self.b_init_lora_k + residuals_lora_k[:, self.embed_dim:].T
            self.a_proj_weight_lora_v = self.a_init_lora_v + residuals_lora_v[:, :self.embed_dim]
            self.b_proj_weight_lora_v = self.b_init_lora_v + residuals_lora_v[:, self.embed_dim:].T

    
    def _reset_lora(self):
        
        kaiming_uniform_(self.a_proj_weight_lora_q, a=math.sqrt(5))
        zeros_(self.b_proj_weight_lora_q)
        kaiming_uniform_(self.a_proj_weight_lora_k, a=math.sqrt(5))
        zeros_(self.b_proj_weight_lora_k)
        kaiming_uniform_(self.a_proj_weight_lora_v, a=math.sqrt(5))
        zeros_(self.b_proj_weight_lora_v)

    def _reset_init_lora(self):

        kaiming_uniform_(self.a_init_lora_q, a=math.sqrt(5))
        zeros_(self.b_init_lora_q)
        kaiming_uniform_(self.a_init_lora_k, a=math.sqrt(5))
        zeros_(self.b_init_lora_k)
        kaiming_uniform_(self.a_init_lora_v, a=math.sqrt(5))
        zeros_(self.b_init_lora_v)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, lora_w: Optional[Tensor] = None, idx=-1) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not self.added_lora:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
            else:
                if lora_w and len(lora_w[0])>idx and self.shared:
                    text_a_q = lora_w[0][idx][0]
                    text_b_q = lora_w[0][idx][1]
                    text_a_k = lora_w[0][idx][2]
                    text_b_k = lora_w[0][idx][3]
                    text_a_v = lora_w[0][idx][4]
                    text_b_v = lora_w[0][idx][5]

                    residuals_lora_q = self.text_vis_proj(torch.concatenate((text_a_q, text_b_q.T)).reshape(self.rank, -1))
                    residuals_lora_k = self.text_vis_proj(torch.concatenate((text_a_k, text_b_k.T)).reshape(self.rank, -1))
                    residuals_lora_v = self.text_vis_proj(torch.concatenate((text_a_v, text_b_v.T)).reshape(self.rank, -1))

                    self.a_proj_weight_lora_q = self.a_init_lora_q + residuals_lora_q[:, :self.embed_dim]
                    self.b_proj_weight_lora_q = self.b_init_lora_q + residuals_lora_q[:, self.embed_dim:].T
                    self.a_proj_weight_lora_k = self.a_init_lora_k + residuals_lora_k[:, :self.embed_dim]
                    self.b_proj_weight_lora_k = self.b_init_lora_k + residuals_lora_k[:, self.embed_dim:].T
                    self.a_proj_weight_lora_v = self.a_init_lora_v + residuals_lora_v[:, :self.embed_dim]
                    self.b_proj_weight_lora_v = self.b_init_lora_v + residuals_lora_v[:, self.embed_dim:].T

                attn_output, attn_output_weights = lora_multi_head_attention_forward(
                query=query, key=key, value=value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias,
                bias_k=self.bias_k, bias_v=self.bias_v, add_zero_attn=self.add_zero_attn,
                dropout_p=self.dropout, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, 
                a_proj_weight_lora_q=self.a_proj_weight_lora_q, b_proj_weight_lora_q=self.b_proj_weight_lora_q, 
                a_proj_weight_lora_k=self.a_proj_weight_lora_k, b_proj_weight_lora_k=self.b_proj_weight_lora_k, 
                a_proj_weight_lora_v=self.a_proj_weight_lora_v, b_proj_weight_lora_v=self.b_proj_weight_lora_v, 
                training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, lora=self.added_lora, rank=self.rank, alpha=self.alpha)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            if query.shape[-1]==512 and self.added_lora:
                return attn_output, attn_output_weights, self.a_proj_weight_lora_q, self.b_proj_weight_lora_q, self.a_proj_weight_lora_k, self.b_proj_weight_lora_k, self.a_proj_weight_lora_v, self.b_proj_weight_lora_v
                # return attn_output, attn_output_weights, self.a_proj_weight_lora_q, self.b_proj_weight_lora_q, self.a_proj_weight_lora_v, self.b_proj_weight_lora_v
            else:
                return attn_output, attn_output_weights


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): the input feature.
        """
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Wrapper of GELU activation layer."""

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): the input feature.
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Attention block with residual connection."""

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttentionLoRA(embed_dim=d_model, num_heads=n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.mask_pre_mlp = True

        self.visual_feature = torch.empty(0)
        self.text_feature = torch.empty(0)

    def attention(self, x: torch.Tensor, lora_w=False, idx=-1):
        """Calculate mask multi-head-attention."""
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        if x.shape[-1]==512 and self.attn.added_lora:
            outputs = self.attn(
                x, x, x, need_weights=False, attn_mask=self.attn_mask)
        elif x.shape[-1]==768 and self.attn.added_lora:
            outputs = self.attn(
                x, x, x, need_weights=False, attn_mask=self.attn_mask, lora_w=lora_w, idx=idx)
        else:
            outputs = self.attn(
                x, x, x, need_weights=False, attn_mask=self.attn_mask)
            
        return outputs

    def forward(self, x: torch.Tensor, lora=False, idx=-1):
        """
        Args:
            x (torch.Tensor): the input feature.
        """
        if x.shape[-1]==768:
            self.visual_feat = x
        
        if isinstance(lora, bool) and lora:
            outputs = self.attention(self.ln_1(x), lora_w=lora)
            x = x + outputs[0]
            lora_w = outputs[2:]
        elif isinstance(lora, list):
            x = x + self.attention(self.ln_1(x), lora_w=lora, idx=idx)[0]
            lora_w = None
        else:
            x = x + self.attention(self.ln_1(x))[0]
            lora_w = None

        x = x + self.mlp(self.ln_2(x))
        return x, lora_w

    def forward_dense(self, x: torch.Tensor):
        """Reinplementation of forward function for dense prediction of image
        encoder in CLIP model.

        Args:
            x (torch.Tensor): the input feature.
        """
        if x.shape[-1]==768:
            self.visual_feat = x
        
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        L, N, D = y.shape  # L N 3D

        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0,
                                               3).reshape(3 * N, L, D // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)

        q, k, v = y.tensor_split(3, dim=0)
        v = v.transpose(1, 0) + x  # L N D

        v = v + self.mlp(self.ln_2(v))
        return v

class Transformer(nn.Module):
    """General Transformer Architecture for both image and text encoder."""

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 lora_layers: list,
                 attn_mask: torch.Tensor = None,
                 prompt_length=0,
                 prompt_depth=0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

        self.prompt_length = prompt_length
        self.prompt_depth = prompt_depth
        self.prompt_tokens = nn.Parameter(
            torch.zeros(prompt_depth, prompt_length,
                        width)) if prompt_length > 0 else None
        if self.prompt_tokens is not None:
            nn.init.xavier_uniform_(self.prompt_tokens)

        self.lora_layers = lora_layers
        self.lora_w = []

    def forward(self, x: torch.Tensor, dense=False):
        """
        Args:
            x (torch.Tensor): input features.
            dense (bool): whether use reimplemented dense forward
                function in the last layer.
        """
        lora_w = []
        for i, resblock in enumerate(self.resblocks):
            if self.prompt_length > 0 and i < self.prompt_depth:
                length = self.prompt_length + 1 if i > 0 else 1
                x = torch.cat((x[0:1, :, :], self.prompt_tokens[i].repeat(
                    x.shape[1], 1, 1).permute(1, 0, 2), x[length:, :, :]))

            if i == self.layers - 1 and dense:
                x = resblock.forward_dense(x)
                x = torch.cat((x[0:1, :, :], x[self.prompt_length + 1::, :]),
                              dim=0)
            else:
                if i in self.lora_layers and self.width==512:
                    x, lora_w_i = resblock(x, lora=True)
                    lora_w.append(lora_w_i)
                elif i in self.lora_layers and self.width==768:
                    x, _ = resblock(x, lora=self.lora_w, idx=i)
                else:
                    x, _ = resblock(x)

        if len(lora_w)!=0 and len(self.lora_w)<1:
            self.lora_w.append(lora_w)

        return x
                
class VisualTransformer(nn.Module):
    """Visual encoder for CLIP model."""

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int, prompt_depth: int,
                 prompt_length: int, lora_layers: list):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            lora_layers=lora_layers,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length,)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.patch_size = patch_size
        self.input_resolution = input_resolution

    def forward(self, x: torch.Tensor, dense=False):
        """
        Args:
            x (torch.Tensor): input features.
            dense (bool): whether use reimplemented dense forward
                function in the last layer.
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        if dense and (x.shape[1] != self.positional_embedding.shape[0]):
            x = x + self.resized_pos_embed(self.input_resolution,
                                           x.shape[1]).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, dense=dense)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if dense:
            x = self.ln_post(x[:, :, :])
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def resized_pos_embed(self, in_res, tgt_res, mode='bicubic'):
        """Resize the position embedding."""
        # assert L == (input_resolution // self.patch_size) ** 2 + 1
        L, D = self.positional_embedding.shape

        in_side = in_res // self.patch_size
        # tgt_side = tgt_res // self.patch_size
        tgt_side = int((tgt_res - 1)**0.5)

        cls_pos = self.positional_embedding[0].unsqueeze(0)  # 1 D
        pos_embed = self.positional_embedding[1:].reshape(
            1, in_side, in_side, D).permute(0, 3, 1, 2)  # L-1 D -> 1 D S S
        resized_pos_embed = F.interpolate(
            pos_embed,
            size=(tgt_side, tgt_side),
            mode=mode,
            align_corners=False,
        )  # 1 D S S -> 1 D S' S'
        resized_pos_embed = resized_pos_embed.squeeze(0).reshape(
            D, -1).T  # L'-1 D

        return torch.cat((cls_pos, resized_pos_embed), dim=0)

class CLIP(nn.Module):
    """Custom implementation of CLIP model.

    Refer to: https://github.com/openai/CLIP
    """

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # prompt
        prompt_depth: int = 0,
        prompt_length: int = 0,
        design_details=None,
    ):
        super().__init__()

        self.context_length = context_length

        self.image_resolution = image_resolution

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            lora_layers=design_details.lora_layers,
            output_dim=embed_dim,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            lora_layers=design_details.lora_layers,
            attn_mask=self.build_attention_mask(),
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

    def build_attention_mask(self):
        """Create causal attention mask."""
        # lazily create causal attention mask, with full attention between
        # the vision tokens pytorch uses additive attention mask; fill with
        # -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        """Return the dtype of the model."""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, masks=None, pool_mask=None, dense=False):
        """Image encoding."""
        if pool_mask is not None:
            return self.visual(
                image.type(self.dtype), mask=pool_mask, dense=dense)
        if masks is None:
            return self.visual(image.type(self.dtype), dense=dense)
        else:
            return self.visual(image.type(self.dtype), masks.type(self.dtype))

    def encode_text(self, text):
        """Texts encoding."""
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)
        x = x[torch.arange(x.shape[0]),
              text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """
        Args:
            image (torch.Tensor): input images.
            text (torch.Tensor): input text.
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # normalized features
        # image_features shape: [1, 1024]
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_iamge = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_iamge, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16."""

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [
                    *[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ['text_projection', 'proj']:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, prompt_depth=0, prompt_length=0, design_details=None):
    """Build a CLIP model from given pretrained weights."""
    vit = 'visual.proj' in state_dict

    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len({
                k.split('.')[2]
                for k in state_dict if k.startswith(f'visual.layer{b}')
            }) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round(
            (state_dict['visual.attnpool.positional_embedding'].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            'visual.attnpool.positional_embedding'].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len({
        k.split('.')[2]
        for k in state_dict if k.startswith('transformer.resblocks')
    })

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        prompt_depth=prompt_depth,
        prompt_length=prompt_length,
        design_details=design_details
    )

    for key in ['input_resolution', 'context_length', 'vocab_size']:
        del state_dict[key]

    convert_weights(model)

    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print('Weights not found for some missing keys: ', missing_keys)

    return model.eval()
