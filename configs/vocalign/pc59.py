import os
PWD = os.getcwd()

_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/pascal_context_59_384x384.py'
]

custom_imports = dict(imports=['cat_seg'])
randomness = {'seed': 42}
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (384, 384)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    n_views=1,
    jitter_aug=True,
    # due to the clip model, we do normalization in backbone forward()
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# model_cfg
model = dict(
    type='UDADecorator',
    model=dict(
        type='EncoderDecoder',
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type='CLIPOVCATSeg',
            feature_extractor=dict(
                type='ResNet',
                depth=101,
                # only use the first three layers
                num_stages=3,
                out_indices=(0, 1, 2),
                dilations=(1, 1, 1),
                strides=(1, 2, 2),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
                init_cfg=dict(
                    type='Pretrained', checkpoint='torchvision://resnet101'),
            ),
            train_class_json=
            f'{PWD}/data/pascal_context_59/pc59.json',
            test_class_json=
            f'{PWD}/data/pascal_context_59/pc59.json',
            clip_pretrained='ViT-B/16',
            prompt_ensemble_type = 'imagenet_select_clip',#'imagenet',#
            clip_finetune='attention',
            design_details=dict(
                lora_layers=[0, 1, 2, 3, 4],
            ),
        ),
        neck=dict(
            type='CATSegAggregator',
            appearance_guidance_dim=1024,
            num_layers=2,
            pooling_size=(1, 1),
            topk_templates = None,
        ),
        decode_head=dict(
            type='CATSegHead',
            in_channels=128,
            channels=128,
            num_classes=59,
            embed_dims=128,
            decoder_dims=(64, 32),
            decoder_guidance_dims=(512, 256),
            decoder_guidance_proj_dims=(32, 16),
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                use_thresh=False,
                loss_weight=1.0,
                avg_non_ignore=True)),
        train_cfg=dict(),
        test_cfg=dict(mode='slide', stride=crop_size, crop_size=crop_size),
    ),
    vocalign=dict(masking=True,
        random=False,
        linear=False,
        class_guidance=True,
        classes_to_concepts={
            0: [0, 1, 2],
            1: [3, 4, 5, 6, 7],
            2: [8, 9],
            3: [10, 11, 12, 13],
            4: [14, 15],
            5: [16, 17],
            6: [18, 19],
            7: [20, 21],
            8: [22, 23],
            9: [24, 25],
            10: [26, 27, 28, 29, 30],
            11: [31, 32],
            12: [33, 34, 35],
            13: [36, 37],
            14: [38, 39],
            15: [40, 41, 42, 43],
            16: [44, 45],
            17: [46, 47, 48],
            18: [49, 50, 51],
            19: [52, 53],
            20: [54, 55, 56],
            21: [57, 58],
            22: [59, 60],
            23: [61, 62, 63],
            24: [64, 65, 66],
            25: [67, 68, 69, 70, 71, 72, 73],
            26: [74, 75, 76],
            27: [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
            28: [92, 93],
            29: [94, 95, 96, 97, 98, 99, 100],
            30: [101, 102],
            31: [103, 104],
            32: [105, 106],
            33: [107, 108, 109],
            34: [110, 111],
            35: [112, 113],
            36: [114, 115],
            37: [116, 117, 118, 119],
            38: [120, 121],
            39: [122, 123, 124],
            40: [125, 126],
            41: [127, 128],
            42: [129, 130],
            43: [131, 132, 133],
            44: [134, 135, 136],
            45: [137, 138, 139, 140],
            46: [141, 142, 143, 144],
            47: [145, 146],
            48: [147, 148, 149, 150],
            49: [151, 152, 153],
            50: [154, 155, 156],
            51: [157, 158],
            52: [159, 160],
            53: [161, 162],
            54: [163, 164, 165],
            55: [166, 167, 168, 169, 170, 171, 172, 173, 174],
            56: [175, 176, 177, 178, 179],
            57: [180, 181, 182, 183],
            58: [184, 185]
        },
        concepts_json=
        f'{PWD}/data/pascal_context_59/pc59_guidance.json',
        latent=False,
        latent_lambda = 0.1,
        mask_alpha=0.99,
        mask_pseudo_threshold=0.968,
        mask_lambda=1.0,
        mask_generator=dict(type='block', mask_ratio=0.5, mask_block_size=16), 
        pseudo_weight_ignore_top=0,
        pseudo_weight_ignore_bottom=0,
        n_views=1,
        topk=35,
        randomk=None,
        use_sample_prompt=False),
    gamma_chi=0.99,
    zeta=0.75,
    beta=1e-2)

# lora settings
lora_layers = [0, 1, 2, 3, 4]
lora_rank = 2
alpha = 2
neck = False
shared = False
lora_eval = False

# dataset settings
train_dataloader = dict(
    batch_size=1,   # default 2, good 4
    num_workers=4,
)

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=15000, val_interval=5000)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=80000),
    visualization=dict(
        type='SegVisualizationHook', draw=True, interval=500000))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.0001),    #base lr=1e-5
    paramwise_cfg=dict(
        custom_keys={
            'backbone.clip_model': dict(lr_mult=0.1),
            'backbone.feature_extractor': dict(lr_mult=0.0),
            'decode_head': dict(lr_mult=0.0),
            'neck': dict(lr_mult=0.0),
            'teacher': dict(lr_mult=0.0)
        }))

# Data parallel
model_wrapper_cfg = dict(
    type='MMDistributedDataParallelUDA'
)

# learning policy
param_scheduler = [
    # Use a linear warm-up at [0, 100) iterations
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type='CosineAnnealingLR',
        T_max=79500,
        by_epoch=False,
        begin=500,
        end=3000),
]