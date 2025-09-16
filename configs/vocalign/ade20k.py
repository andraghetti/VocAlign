import os
PWD = os.getcwd()

_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/ade20k_384x384.py'
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
            f'{PWD}/data/ADE20k/ade20k.json',
            test_class_json=
            f'{PWD}/data/ADE20k/ade20k.json',
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
            topk_templates = 40,
        ),
        decode_head=dict(
            type='CATSegHead',
            in_channels=128,
            channels=128,
            num_classes=150,
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
        classes_to_concepts = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            1: [9, 10, 11],
            2: [12, 13, 14],
            3: [15, 16, 17, 18, 19, 20, 21],
            4: [22, 23],
            5: [24, 25, 26, 27],
            6: [28, 29],
            7: [30, 31],
            8: [32, 33, 34],
            9: [35, 36],
            10: [37, 38],
            11: [39, 40, 41],
            12: [42, 43],
            13: [44, 45, 46],
            14: [47, 48, 49],
            15: [50, 51, 52],
            16: [53, 54],
            17: [55, 56, 57],
            18: [58, 59],
            19: [60, 61],
            20: [62, 63],
            21: [64, 65, 66],
            22: [67, 68, 69],
            23: [70, 71, 72],
            24: [73, 74],
            25: [75, 76],
            26: [77, 78, 79],
            27: [80, 81, 82],
            28: [83, 84, 85],
            29: [86, 87, 88],
            30: [89, 90, 91],
            31: [92, 93, 94],
            32: [95, 96],
            33: [97, 98, 99],
            34: [100, 101],
            35: [102, 103, 104],
            36: [105, 106, 107],
            37: [108, 109, 110, 111],
            38: [112, 113],
            39: [114, 115, 116, 117],
            40: [118, 119, 120],
            41: [121, 122, 123],
            42: [124, 125, 126],
            43: [127, 128, 129],
            44: [130, 131, 132],
            45: [133, 134],
            46: [135, 136],
            47: [137, 138],
            48: [139, 140, 141],
            49: [142, 143, 144],
            50: [145, 146],
            51: [147, 148, 149],
            52: [150, 151, 152],
            53: [153, 154],
            54: [155, 156, 157, 158],
            55: [159, 160, 161],
            56: [162, 163, 164, 165],
            57: [166, 167],
            58: [168, 169, 170, 171],
            59: [172, 173, 174, 175],
            60: [176, 177, 178],
            61: [179, 180],
            62: [181, 182, 183, 184],
            63: [185, 186, 187],
            64: [188, 189, 190, 191],
            65: [192, 193],
            66: [194, 195],
            67: [196, 197],
            68: [198, 199],
            69: [200, 201],
            70: [202, 203, 204],
            71: [205, 206, 207],
            72: [208, 209, 210, 211],
            73: [212, 213, 214, 215],
            74: [216, 217, 218],
            75: [219, 220],
            76: [221, 222],
            77: [223, 224, 225, 226],
            78: [227, 228, 229, 230],
            79: [231, 232, 233, 234],
            80: [235, 236],
            81: [237, 238],
            82: [239, 240],
            83: [241, 242],
            84: [243, 244, 245, 246],
            85: [247, 248, 249, 250],
            86: [251, 252, 253, 254],
            87: [255, 256, 257],
            88: [258, 259, 260, 261],
            89: [262, 263, 264],
            90: [265, 266],
            91: [267, 268, 269, 270],
            92: [271, 272, 273],
            93: [274, 275, 276, 277],
            94: [278, 279, 280],
            95: [281, 282, 283, 284],
            96: [285, 286, 287, 288],
            97: [289, 290, 291],
            98: [292, 293],
            99: [294, 295, 296, 297],
            100: [298, 299],
            101: [300, 301, 302, 303],
            102: [304, 305, 306, 307],
            103: [308, 309, 310],
            104: [311, 312, 313],
            105: [314, 315, 316],
            106: [317, 318, 319, 320],
            107: [321, 322, 323],
            108: [324, 325, 326],
            109: [327, 328, 329, 330],
            110: [331, 332, 333],
            111: [334, 335],
            112: [336, 337],
            113: [338, 339, 340],
            114: [341, 342],
            115: [343, 344, 345],
            116: [346, 347, 348],
            117: [349, 350, 351],
            118: [352, 353],
            119: [354, 355, 356],
            120: [357, 358, 359],
            121: [360, 361],
            122: [362, 363],
            123: [364, 365, 366],
            124: [367, 368],
            125: [369, 370, 371],
            126: [372, 373, 374],
            127: [375, 376],
            128: [377, 378, 379],
            129: [380, 381],
            130: [382, 383],
            131: [384, 385],
            132: [386, 387, 388],
            133: [389, 390],
            134: [391, 392],
            135: [393, 394],
            136: [395, 396],
            137: [397, 398],
            138: [399, 400],
            139: [401, 402],
            140: [403, 404],
            141: [405, 406, 407],
            142: [408, 409],
            143: [410, 411],
            144: [412, 413],
            145: [414, 415],
            146: [416, 417],
            147: [418, 419, 420],
            148: [421, 422],
            149: [423, 424, 425]
        },
        concepts_json=
        f'{PWD}/data/ADE20k/ade20k_guidance.json',
        latent=False,
        latent_lambda = 0.1,
        mask_alpha=0.99,
        mask_pseudo_threshold=0.968,
        mask_lambda=1.0,
        mask_generator=dict(type='block', mask_ratio=0.5, mask_block_size=16), 
        pseudo_weight_ignore_top=0,
        pseudo_weight_ignore_bottom=0,
        n_views=1,
        topk=50,
        randomk=None,
        use_sample_prompt=True),
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
    batch_size=1, 
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