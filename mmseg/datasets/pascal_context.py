# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PascalContextDataset(BaseSegDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        ann_file (str): Annotation file path.
    """

    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file='',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args)


@DATASETS.register_module()
class PascalContextDataset59(BaseSegDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    True. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.
    Noted: If the background is 255 and the ids of categories are from 0 to 58,
    ``reduce_zero_label`` needs to be set to False.

    Args:
        ann_file (str): Annotation file path.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file='',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args)


@DATASETS.register_module()
class PascalContextDataset459(BaseSegDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    True. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.
    Noted: If the background is 255 and the ids of categories are from 0 to 58,
    ``reduce_zero_label`` needs to be set to False.

    Args:
        ann_file (str): Annotation file path.
    """
    METAINFO = dict(
        classes=("accordion", "aeroplane", "airconditioner", "antenna", "artillery",
                "ashtray", "atrium", "babycarriage", "bag", "ball", "balloon", "bambooweaving",
                "barrel", "baseballbat", "basket", "basketballbackboard", "bathtub", "bed",
                "bedclothes", "beer", "bell", "bench", "bicycle", "binoculars", "bird",
                "birdcage", "birdfeeder", "birdnest", "blackboard", "board", "boat", "bone",
                "book", "bottle", "bottleopener", "bowl", "box", "bracelet", "brick",
                "bridge", "broom", "brush", "bucket", "building", "bus", "cabinet",
                "cabinetdoor", "cage", "cake", "calculator", "calendar", "camel", "camera",
                "cameralens", "can", "candle", "candleholder", "cap", "car", "card", "cart",
                "case", "casetterecorder", "cashregister", "cat", "cd", "cdplayer", "ceiling",
                "cellphone", "cello", "chain", "chair", "chessboard", "chicken", "chopstick",
                "clip", "clippers", "clock", "closet", "cloth", "clothestree", "coffee", "coffeemachine",
                "comb", "computer", "concrete", "cone", "container", "controlbooth", "controller",
                "cooker", "copyingmachine", "coral", "cork", "corkscrew", "counter", "court", "cow",
                "crabstick", "crane", "crate", "cross", "crutch", "cup", "curtain", "cushion",
                "cuttingboard", "dais", "disc", "disccase", "dishwasher", "dock", "dog", "dolphin", "door", 
                "drainer", "dray", "drinkdispenser", "drinkingmachine", "drop", "drug", "drum", "drumkit", 
                "duck", "dumbbell", "earphone", "earrings", "egg", "electricfan", "electriciron", "electricpot", 
                "electricsaw", "electronickeyboard", "engine", "envelope", "equipment", "escalator", 
                "exhibitionbooth", "extinguisher", "eyeglass", "fan", "faucet", "faxmachine", "fence", "ferriswheel", 
                "fireextinguisher", "firehydrant", "fireplace", "fish", "fishtank", "fishbowl", "fishingnet", "fishingpole", 
                "flag", "flagstaff", "flame", "flashlight", "floor", "flower", "fly", "foam", "food", "footbridge", 
                "forceps", "fork", "forklift", "fountain", "fox", "frame", "fridge", "frog", "fruit", "funnel", "furnace", 
                "gamecontroller", "gamemachine", "gascylinder", "gashood", "gasstove", "giftbox", "glass", "glassmarble", 
                "globe", "glove", "goal", "grandstand", "grass", "gravestone", "ground", "guardrail", "guitar", "gun", "hammer", 
                "handcart", "handle", "handrail", "hanger", "harddiskdrive", "hat", "hay", "headphone", "heater", "helicopter", 
                "helmet", "holder", "hook", "horse", "horse-drawncarriage", "hot-airballoon", "hydrovalve", "ice", "inflatorpump", 
                "ipod", "iron", "ironingboard", "jar", "kart", "kettle", "key", "keyboard", "kitchenrange", "kite", "knife", 
                "knifeblock", "ladder", "laddertruck", "ladle", "laptop", "leaves", "lid", "lifebuoy", "light", "lightbulb", 
                "lighter", "line", "lion", "lobster", "lock", "machine", "mailbox", "mannequin", "map", "mask", "mat", 
                "matchbook", "mattress", "menu", "metal", "meterbox", "microphone", "microwave", "mirror", "missile", 
                "model", "money", "monkey", "mop", "motorbike", "mountain", "mouse", "mousepad", "musicalinstrument", 
                "napkin", "net", "newspaper", "oar", "ornament", "outlet", "oven", "oxygenbottle", "pack", "pan", "paper", 
                "paperbox", "papercutter", "parachute", "parasol", "parterre", "patio", "pelage", "pen", "pencontainer", "pencil", 
                "person", "photo", "piano", "picture", "pig", "pillar", "pillow", "pipe", "pitcher", "plant", "plastic", "plate", 
                "platform", "player", "playground", "pliers", "plume", "poker", "pokerchip", "pole", "pooltable", "postcard", 
                "poster", "pot", "pottedplant", "printer", "projector", "pumpkin", "rabbit", "racket", "radiator", "radio", 
                "rail", "rake", "ramp", "rangehood", "receiver", "recorder", "recreationalmachines", "remotecontrol", "road", 
                "robot", "rock", "rocket", "rockinghorse", "rope", "rug", "ruler", "runway", "saddle", "sand", "saw", "scale", 
                "scanner", "scissors", "scoop", "screen", "screwdriver", "sculpture", "scythe", "sewer", "sewingmachine", "shed", 
                "sheep", "shell", "shelves", "shoe", "shoppingcart", "shovel", "sidecar", "sidewalk", "sign", "signallight", 
                "sink", "skateboard", "ski", "sky", "sled", "slippers", "smoke", "snail", "snake", "snow", "snowmobiles", "sofa", 
                "spanner", "spatula", "speaker", "speedbump", "spicecontainer", "spoon", "sprayer", "squirrel", "stage", "stair", 
                "stapler", "stick", "stickynote", "stone", "stool", "stove", "straw", "stretcher", "sun", "sunglass", "sunshade", 
                "surveillancecamera", "swan", "sweeper", "swimring", "swimmingpool", "swing", "switch", "table", "tableware", "tank", 
                "tap", "tape", "tarp", "telephone", "telephonebooth", "tent", "tire", "toaster", "toilet", "tong", "tool", 
                "toothbrush", "towel", "toy", "toycar", "track", "train", "trampoline", "trashbin", "tray", "tree", "tricycle", 
                "tripod", "trophy", "truck", "tube", "turtle", "tvmonitor", "tweezers", "typewriter", "umbrella", "unknown", 
                "vacuumcleaner", "vendingmachine", "videocamera", "videogameconsole", "videoplayer", "videotape", "violin", 
                "wakeboard", "wall", "wallet", "wardrobe", "washingmachine", "watch", "water", "waterdispenser", "waterpipe", 
                "waterskateboard", "watermelon", "whale", "wharf", "wheel", "wheelchair", "window", "windowblinds", "wineglass", 
                "wire", "wood", "wool"),
        palette=[])

    def __init__(self,
                 ann_file='',
                 img_suffix='.jpg',
                 seg_map_suffix='.tif',
                 reduce_zero_label=True,
                 ignore_index=65535,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args)