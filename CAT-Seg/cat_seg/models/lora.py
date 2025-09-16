# Sheng Wang at Feb 22 2023
import torch.nn as nn
from torch.nn.init import zeros_


class projectionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, bias):
        super().__init__()
        
        self.linear1 = nn.Linear(in_features=input_dim, out_features=output_dim//4, bias=bias)
        self.linear2 = nn.Linear(in_features=output_dim//4, out_features=output_dim, bias=bias)

        self.gelu = nn.GELU()
        self._init_linear()
    
    def _init_linear(self):
        zeros_(self.linear1.weight)
        zeros_(self.linear2.weight)

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.gelu(self.linear2(x))

        return x


class LoRA(nn.Module):
    """Applies low-rank adaptation to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self,
                 model,
                 device,
                 r: int,
                 lora_layers=None,
                 shared=False,
                 alpha=1):
        super(LoRA, self).__init__()
        assert r > 0
        
        if lora_layers:
            self.lora_layers = lora_layers
        else:
            self.lora_layers = list(range(len(model.backbone.clip_model.visual.transformer.resblocks)))
        
        model.backbone.clip_model.transformer.lora_layers = self.lora_layers

        # let's freeze first
        for param in model.backbone.clip_model.transformer.parameters():
            param.requires_grad = False
        
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(
                model.backbone.clip_model.transformer.resblocks):
            # If we only want few lora layer instead of all
            if t_layer_i in self.lora_layers:
                blk.attn._add_lora(device=device, shared=False, r=r, alpha=alpha)

        # lets freeze first
        for param in model.backbone.clip_model.visual.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(
                model.backbone.clip_model.visual.transformer.resblocks):
            # If we only want few lora layer instead of all
            if t_layer_i in self.lora_layers:
                if shared:
                    text_vis_proj = projectionMLP(2*model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.out_proj.out_features, 2*blk.attn.out_proj.out_features, bias=False).to(device)

                    text_a_q = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.a_proj_weight_lora_q
                    text_b_q = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.b_proj_weight_lora_q
                    text_a_k = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.a_proj_weight_lora_k
                    text_b_k = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.b_proj_weight_lora_k
                    text_a_v = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.a_proj_weight_lora_v
                    text_b_v = model.backbone.clip_model.transformer.resblocks[t_layer_i].attn.b_proj_weight_lora_v
                    
                    text_lora= [text_a_q, text_b_q, text_a_k, text_b_k, text_a_v, text_b_v]
                    
                    blk.attn._add_lora(device=device, shared=shared, r=r, text_vis_proj=text_vis_proj, text_lora=text_lora, alpha=alpha)
                else:
                    blk.attn._add_lora(device=device, shared=shared, r=r, alpha=alpha)
