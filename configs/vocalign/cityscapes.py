import os
PWD = os.getcwd()

_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/cityscapes_384x384.py'
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
            f'{PWD}/data/cityscapes/cityscapes.json',
            test_class_json=
            f'{PWD}/data/cityscapes/cityscapes.json',
            clip_pretrained='ViT-B/16',
            prompt_ensemble_type = 'imagenet_select_clip',#'imagenet',#
            clip_finetune='attention',
            design_details=dict(
                lora_layers=[0, 1, 2, 3, 4],
                lora_rank=1,
                alpha = 1,
                neck = False,
                shared = False,
            ),
        ),
        neck=dict(
            type='CATSegAggregator',
            appearance_guidance_dim=1024,
            num_layers=2,
            pooling_size=(1, 1),
            topk_templates = None,  # How many topk templates
        ),
        decode_head=dict(
            type='CATSegHead',
            in_channels=128,
            channels=128,
            num_classes=19,
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
    vocalign=dict(
        lora_eval=False,
        masking=True,   # This corresponds to using masking
        random=False,
        linear=False,
        class_guidance=True,   # Vocabulary alignment
        classes_to_concepts={
            0: [0],
            1: [1],
            2: [2, 3],
            3: [4],
            4: [5, 6],
            5: [7, 8, 9, 10],
            6: [11, 12],
            7: [13, 14, 15, 16],
            8: [17, 18],
            9: [19, 20, 21, 22],
            10: [23],
            11: [24, 25],
            12: [26, 27, 28],
            13: [29, 30],
            14: [31, 32],
            15: [33],
            16: [34, 35],
            17: [36, 37],
            18: [38]
        },
        concepts_json=
        f'{PWD}/data/cityscapes/cityscapes_guidance.json',
        latent=False,
        latent_lambda = 0.1,
        mask_alpha=0.99,
        mask_pseudo_threshold=0.968,
        mask_lambda=1.0,
        mask_generator=dict(type='block', mask_ratio=0.7, mask_block_size=16), #mask ratio=0.7
        pseudo_weight_ignore_top=0,
        pseudo_weight_ignore_bottom=0,
        n_views=1,
        topk=19,  # Topk classes
        randomk=None,   #RandomK classes
        use_sample_prompt=False #TopK prompt templates
        ),
    gamma_chi=0.99,
    zeta=0.75,
    beta=1e-2)

# dataset settings
train_dataloader = dict(
    batch_size=2,   #2
    num_workers=4,
)

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=5000) #40000 5000

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=80000),
    visualization=dict(
        type='SegVisualizationHook', draw=True, interval=80000))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.0001),    #base lr 5e-5
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
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=100),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type='CosineAnnealingLR',
        T_max=79500,
        by_epoch=False,
        begin=500,
        end=3000),
]
