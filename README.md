<div align="center">
  <img src="resources/results.png"/>
  <div>&nbsp;</div>
  </div>
  <div>&nbsp;</div>

# VocAlign: Source-free Domain Adaptation for Open-vocabulary Semantic Segmentation

[![Docker Release](https://github.com/Sisso16/VocAlign/actions/workflows/docker.yml/badge.svg)](https://github.com/Sisso16/VocAlign/actions/workflows/docker.yml)

This is the official implementation of VocAlign, a method to perform source-free domain adaptation on CAT-Seg. 
VocAlign is presented in **"Lost in Translation? Vocabulary Alignment for Source-Free Adaptation in Open-Vocabulary Semantic Segmentation"**, [BMVC 2025](https://bmvc2025.bmva.org/)
If you find this code useful, please cite our work:

```
@inproceedings{Mazzucco2025lost,
  title        = {Lost in Translation? Vocabulary Alignment for Source-Free Domain Adaptation in Open-Vocabulary Semantic Segmentation},
  author       = {Mazzucco, Silvio and Persson, Carl and Segu, Mattia and Dovesi, Pier Luigi and Tombari, Federico and Van Gool, Luc and Poggi, Matteo},
  booktitle    = {British Machine Vision Conference},
  note         = {BMVC},
  year         = {2025}
}
```

### Installation

**Step 1.** Clone the repository with recursive option for the submodule:

```shell
git clone --recursive https://github.com/Sisso16/VocAlign
```

If you already cloned without `--recursive`, initialize the submodule:

```shell
git submodule update --init --recursive
```

**Step 2.** Create conda environment and install packages:

```shell
conda env create -f environment.yml
conda run -n vocalign mim install "mmcv==2.1.0"
conda activate vocalign
```


### Dataset

```none
VocAlign
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── cityscapes.json
│   │   ├── cityscapes_guidance.json
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```
First download cityscapes dataset (leftImg8bit) and ground truth (gtFine) from their [website](https://www.cityscapes-dataset.com/). 
Then in the `data/cityscapes` directory add the `cityscapes.json` and `cityscapes_guidance.json` files.

cityscapes.json is the following:

```none
["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
```

cityscapes_guidance.json used in our experiments to achieve the best result was:

```none
["road", "sidewalk", "building", "house", "individual standing wall which is not part of a building", "fence", "hole in fence", "pole", "metal", "railing", "sign pole", "traffic light", "light", "traffic sign", "street sign", "parking sign", "direction sign", "vegetation", "tree", "terrain", "grass", "soil", "sand", "sky", "person", "pedestrian", "rider", "driver", "passenger", "car", "van", "truck", "truck trailer", "bus", "train", "tram", "motorcycle", "scooter", "bicycle"]
```

If you modify the concepts in `cityscapes_guidance` make sure you also modify the `classes_to_concepts` dict(list) assignments in the correspondent config file.

### Run experiments
To resume from pre-trained CAT-Seg, create a folder `work_dirs/cityscapes/`, in this folder create a file `last_checkpoint` and copy paste the path to the checkpoint. The link to download the checkpoint can be found in the README within the `CAT-Seg/` directory.


#### Train VocAlign
```shell
vocalign-train configs/vocalign/cityscapes.py --resume
```

#### Evaluate VocAlign
In the config file set `lora_eval` to True. In `last_checkpoint` the path automatically updates at the end of VocAlign training. If you just want to perform inference without previously having conducted the training yourself, just copy paste manually the path to the desired VocAlign checkpoint.

```shell
vocalign-test configs/vocalign/cityscapes.py work_dirs/cityscapes/best_cityscapes.pth
```

If you want to visualize results generated during evaluation, specify the work_dir and change the `interval` in the `SegVisualizationHook` within the config file before launching the evaluation.

### Docker Installation (Alternative)

For a quick start or if you prefer containerized environments, you can use our pre-built Docker image:

```shell
# Train VocAlign
docker run --platform linux/amd64 --rm --gpus all \
  -v $(git rev-parse --show-toplevel):/workspace \
  ghcr.io/sisso16/vocalign:latest \
  vocalign-train configs/vocalign/cityscapes.py --resume

# Evaluate VocAlign
docker run --platform linux/amd64 --rm --gpus all \
  -v $(git rev-parse --show-toplevel):/workspace \
  ghcr.io/sisso16/vocalign:latest \
  vocalign-test configs/vocalign/cityscapes.py work_dirs/cityscapes/best_cityscapes.pth
```

### Other
We based our code on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) codebase.
