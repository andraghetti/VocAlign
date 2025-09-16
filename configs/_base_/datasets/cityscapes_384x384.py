import os
PWD = os.getcwd()

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = f'{PWD}/data/cityscapes'
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline,
        label_map={
            0: 255,
            1: 255,
            2: 255,
            3: 255,
            4: 255,
            5: 255,
            6: 255,
            7: 0,
            8: 1,
            9: 255,
            10: 255,
            11: 2,
            12: 3,
            13: 4,
            14: 255,
            15: 255,
            16: 255,
            17: 5,
            18: 255,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 255,
            30: 255,
            31: 16,
            32: 17,
            33: 18,
            -1: 255
        }))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline,
        label_map={
            0: 255,
            1: 255,
            2: 255,
            3: 255,
            4: 255,
            5: 255,
            6: 255,
            7: 0,
            8: 1,
            9: 255,
            10: 255,
            11: 2,
            12: 3,
            13: 4,
            14: 255,
            15: 255,
            16: 255,
            17: 5,
            18: 255,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 255,
            30: 255,
            31: 16,
            32: 17,
            33: 18,
            -1: 255
        }))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
