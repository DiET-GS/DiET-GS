#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs

from gaussian_renderer import render

import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr

import pyiqa

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    clipiqa_test = 0.0
    musiq_test = 0.0
    
    clipiqa = pyiqa.create_metric('clipiqa').cuda()
    musiq = pyiqa.create_metric('musiq').cuda()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if name == "train":
            continue
        
        rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], 0.0, 1.0)
        gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0)

        if name == "test":
            psnr_test += psnr(rendering, gt).mean().double()
            ssim_test += ssim(rendering, gt).mean().double()
            lpips_test += lpips(rendering, gt).mean().double()
            clipiqa_test += clipiqa(rendering.unsqueeze(0))
            musiq_test += musiq(rendering.unsqueeze(0))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if name == "test":
        lpips_test /= len(views)
        psnr_test /= len(views)
        ssim_test /= len(views)
        clipiqa_test /= len(views)
        musiq_test /= len(views)

        print("  SSIM : {}".format(ssim_test))
        print("  PSNR : {}".format(psnr_test))
        print("  LPIPS: {}".format(lpips_test))
        print("  MUSIQ: {}".format(musiq_test))
        print("  CLIP-IQA: {}".format(clipiqa_test))
        print("")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)