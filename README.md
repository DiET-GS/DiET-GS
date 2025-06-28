<p align="center">
  <h1 align="center">DiET-GS 🫨 <br>
Diffusion Prior and Event Stream-Assisted <br>
Motion Deblurring 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/seungjun-lee-43101a261/">Seungjun Lee</a></span> ·  
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup></sup> <br>
    Department of Computer Science, National University of Singapore<br>
  </p>
  <h2 align="center">CVPR 2025</h2>
  <h3 align="center"><a href="https://github.com/DiET-GS/DiET-GS">Code</a> | <a href="https://arxiv.org/abs/2503.24210">Paper</a> | <a href="https://diet-gs.github.io">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  </div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/DiET-GS/DiET-GS/blob/main/asset/teaser.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
Our <strong>DiET-GS++</strong> enables high quality novel-view synthesis with recovering precise color and well-defined details from the blurry multi-view images.
</p>
</p>

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#per-scene-optimization-of-diet-gs-stage-1">Per-scene Optimiazation of DiET-GS (Stage 1)</a>
    </li>
     <li>
      <a href="#per-scene-optimization-of-diet-gs-stage-2">Per-scene Optimiazation of DiET-GS++ (Stage 2)</a>
    </li>
    <li>
      <a href="#render">Render</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News:

- [2025/02/27] DiET-GS is accepted to CVPR 2025 🔥. The code will be released at early June.
- [2025/06/27] The code of DiET-GS 🫨 is released 👊🏻! Now you can train DiET-GS and render the clean images.

## TODO
- [x] Release the code of DiET-GS
- [ ] Release the code of DiET-GS++

## Installation

You can set up a conda environment as follows:
```

```

## Data Preparation

We provide the two pre-processed data:
- <a href="https://huggingface.co/datasets/onandon/DiET-GS/tree/main/ev-deblurnerf_blender">EvDeblur-blender</a>
- <a href="https://huggingface.co/datasets/onandon/DiET-GS/tree/main/ev-deblurnerf_cdavis">EvDeblur-CDAVIS</a>

Above dataset was originally proposed by <a href="https://github.com/uzh-rpg/evdeblurnerf">this work</a>. In our work, we discard the provided ground-truth camera poses for multi-view images, as we assume such information is readily available in real-world scenarios, even when the images exhibit severe motion blur.

To calibrate the camera poses of blurry multi-view images and construct the initial point clouds for 3D Gaussian Splatting (3DGS), we follow a two-step process:

1. Deblur the blurry images with EDI processing.
2. Feed EDI-deblurred multi-view images from step 1 to COLMAP, initializing the 3DGS.

We also provide the toy example for EDI in <a href="https://github.com/DiET-GS/DiET-GS/blob/main/deblur_w_edi.ipynb">deblur_w_edi.ipynb</a>.

You can download the calibrated camera poses and initial point clouds for all scenes in the dataset by running the code below.
```
python download_data.py
```
Note that the script above may also download additional files required for processing event streams during scene optimization.

Once you run the above command, the downloaded files may be located to designated path. Refer to the file structure:
```
data
├── ev-deblurnerf_cdavis   <- Real-world dataset
│   ├── blurbatteries      <- Scene name
│   │   ├── events         <- Event files
│   │   ├── images         <- N blurry multi-view images + 5 ground-truth clean images
│   │   ├── images_edi     <- EDI-deblurred images (Just for COLMAP to initilize 3DGS)
│   │   └── sparse         <- Initial point clouds and camera poses
│   ├── blurfligures 
│   └── ...
│ 
├── ev-deblurnerf_blender   <- Synthetic dataset
│   ├── blurfactory
│   │   ├── events
│   │   ├── images
│   │   ├── images_edi
│   │   └── sparse
│   ├── bluroutdoorpool
│   └── ...
```

## Per-scene Optimization of DiET-GS (Stage 1)

Training DiET-GS without RSD loss on real-world dataset (fast training):
```
SCENE_NAME=blurbatteries

python train_dietgs_wo_rsd_real.py -s data/ev-deblurnerf_cdavis/${SCENE_NAME} --eval -m output/${SCENE_NAME} --event True --event_cmap color --edi_cmap gray --intensity True --edi_simul True --port 6035
```

Training DiET-GS without RSD loss on blender dataset (fast training):
```
SCENE_NAME=blurfactory

python train_dietgs_wo_blender.py -s data/ev-deblurnerf_blender/${SCENE_NAME} --eval -m output/${SCENE_NAME} --event True --event_cmap gray --edi_cmap gray --intensity True --edi_simul True --port 6035
```

Training DiET-GS with RSD loss on real-world dataset:
```
SCENE_NAME=blurbatteries

python train_dietgs_real.py -s data/ev-deblurnerf_cdavis/${SCENE_NAME} --eval -m output/${SCENE_NAME} --event True --event_cmap color --edi_cmap gray --intensity True --edi_simul True --port 6035
```

Training DiET-GS with RSD loss on blender dataset:
```
SCENE_NAME=blurfactory

python train_dietgs_blender.py -s data/ev-deblurnerf_blender/${SCENE_NAME} --eval -m output/${SCENE_NAME} --event True --event_cmap gray --edi_cmap gray --intensity True --edi_simul True --port 6042
```

📌 Note that we set the total iterations to 150000. However, DiET-GS usually converges to optimal performance between 40000-50000 iterations.

## Per-scene Optimization of DiET-GS++ (Stage 2)

*The code for additional optimization of DiET-GS++ (Stage 2) will be released soon!*

## Render

After the scene optimization, you can render the clean images. Specify the iteration number of the pretrained 3DGS model you wish to use.
```
SCENE_NAME=blurbatteries

python render.py -m output/${SCENE_NAME} -s data/ev-deblurnerf_cdavis/${SCENE_NAME} --iteration 50000
```

We also provide pretrained 3DGS in `pretrained/` folder. You can use 3DGS in this folder:
```
SCENE_NAME=blurbatteries

python render.py -m pretrained/ev-deblurnerf_cdavis/${SCENE_NAME} -s data/ev-deblurnerf_cdavis/${SCENE_NAME} --iteration 50000
```

## Acknowledgement

Our work is inspired a lot from the following works. We sincerely appreciate to their great contributions!

* <a href="https://github.com/knelk/enerf">E-NeRF</a>
* <a href="https://wengflow.github.io/robust-e-nerf/">Robust E-NeRF</a>
* <a href="https://github.com/iCVTEAM/E2NeRF">E2NeRF</a>
* <a href="https://github.com/uzh-rpg/EvDeblurNeRF">Ev-DeblurNeRF</a>
* <a href="https://github.com/leejielong/DiSR-NeRF">DiSR-NeRF</a>

## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{lee2025diet,
  title={DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting},
  author={Lee, Seungjun and Lee, Gim Hee},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21739--21749},
  year={2025}
}
```
