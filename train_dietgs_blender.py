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

import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np 
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

from gaussian_renderer import render, network_gui

import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from network.crf import CRF
from event_helpers_loss import *
from utils.event_utils import interpolate_poses, color_event_map_func

import threestudio
from utils.config import load_config
from utils.data import sample_patch

RGBGRAY = torch.tensor([0.299,0.587,0.114])

def training(
    dataset, 
    opt, 
    pipe, 
    testing_iterations, 
    saving_iterations, 
    checkpoint_iterations, 
    checkpoint, 
    debug_from, 
    event, 
    event_cmap,
    edi_cmap,
    edi_simul,
    intensity
    ):
    first_iter = 0

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    cfg = load_config("config/dietgs.yaml")
    
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(cfg.system.prompt_processor)
    prompt_processor.configure_text_encoder()
    prompt_processor.destroy_text_encoder()
    prompt_processor_output = prompt_processor()

    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    guidance.vae.enable_tiling()
    for p in guidance.parameters():
        p.requires_grad=False

    # Loading event data
    if event:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rgbgray = RGBGRAY.to(device)
        
        with open(os.path.join(dataset.source_path, "events/events.pickle"), "rb") as f:
            events = pickle.load(f).to(device)
        with open(os.path.join(dataset.source_path, "events/image_start.pickle"), "rb") as f:
            img_start_stamp = pickle.load(f)
        with open(os.path.join(dataset.source_path, "events/image_end.pickle"), "rb") as f:
            img_end_stamp = pickle.load(f)
        with open(os.path.join(dataset.source_path, "events/id_to_coords.pickle"), "rb") as f:
            id_to_coords = pickle.load(f).to(device)
        with open(os.path.join(dataset.source_path, "events/events_edi_window.pickle"), "rb") as f:
            events_edi_window = torch.from_numpy(pickle.load(f)).to(device)
        id_to_color_map = None
        
        all_tms = []
        for tms_start, tms_end in zip(img_start_stamp, img_end_stamp):
            all_tms.append(np.linspace(tms_start, tms_end, 9))
        all_tms = torch.tensor(np.concatenate(all_tms)).to(device)
        
        interval = (all_tms[1] - all_tms[0]) / 10
        
        all_timestamps = []
        for tms_start, tms_end in zip(img_start_stamp, img_end_stamp):
            all_timestamps.append(torch.from_numpy(np.linspace(tms_start, tms_end, 9)).to(device))
        
        all_poses = []
        train_cameras = scene.getTrainCameras().copy()
        train_cameras_sorted = sorted({int(c.image_name): c for c in train_cameras}.items())

        for _, tc in train_cameras_sorted:
            poses = []
            for tcm in tc.moving_poses:
                pose = np.concatenate([tcm.R, tcm.T[:, None]], axis=-1)
                poses.append(pose)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(device)
            all_poses.append(poses)
            
        crf = gaussians.crf.to(device)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
       
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        if iteration > -1 and (iteration-1) % 160 == 0: # update by epoch 
            guidance.timestep_annealing((iteration-1) // 16)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((4), device="cuda") if opt.random_background else background

        # Simulating blurring process
        all_images = []
        for i, c in enumerate(viewpoint_cam.moving_poses):
            render_pkg = render(c, gaussians, pipe, bg) # NOTE Rendering image with gaussian

            if i == 0:
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            else:
                image = render_pkg["render"]
            all_images.append(image)
        image = torch.stack(all_images, dim=0).mean(dim=0)

        image_id = int(viewpoint_cam.image_name)
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        if event:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, scene.image_h-1, scene.image_h), torch.linspace(0, scene.image_w-1, scene.image_w)),-1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])
            select_inds = np.random.choice(coords.shape[0], size=[opt.sample_pnum], replace=False)
            select_coords = coords[select_inds].long()
            
            all_images_crf = []
            for i in all_images:
                all_images_crf.append(crf(i, skip_learn=iteration<opt.warmup))
            
            event_start_stamp = random.randint(int(img_start_stamp[image_id]), int(img_end_stamp[image_id] - interval))
                
            event_start_idx = torch.searchsorted(events[:, 1], torch.tensor([event_start_stamp]).to(device))
            event_end_idx = torch.searchsorted(events[:, 1], torch.tensor([event_start_stamp + interval]).to(device), side="right")
                
            event_start_pose = interpolate_poses(torch.tensor([event_start_stamp]), all_timestamps[image_id].cpu(), all_poses[image_id].cpu())
            event_end_pose = interpolate_poses(torch.tensor([event_start_stamp + interval]), all_timestamps[image_id].cpu(), all_poses[image_id].cpu())
        
            event_start_camera = Camera(
                colmap_id = -1, 
                R = event_start_pose[0, :3, :3], 
                T = event_start_pose[0, :3, 3], 
                FoVx = scene.getTrainCameras()[0].FoVx, 
                FoVy = scene.getTrainCameras()[0].FoVy, 
                image=None, 
                gt_alpha_mask=None,
                image_name=None, 
                uid=-1, 
                moving_poses=None, 
                data_device=device
            )
                
            event_end_camera = Camera(
                colmap_id = -1, 
                R = event_end_pose[0, :3, :3], 
                T = event_end_pose[0, :3, 3], 
                FoVx = scene.getTrainCameras()[0].FoVx, 
                FoVy = scene.getTrainCameras()[0].FoVy, 
                image=None, 
                gt_alpha_mask=None,
                image_name=None, 
                uid=-1, 
                moving_poses=None, 
                data_device=device
            )
            
            event_start_camera.image_width = image.shape[2]
            event_start_camera.image_height =  image.shape[1]
            event_end_camera.image_width = image.shape[2]
            event_end_camera.image_height = image.shape[1]
                
            event_start_intensity = crf(render(event_start_camera, gaussians, pipe, bg)["render"], skip_learn=iteration<opt.warmup)
            event_end_intensity = crf(render(event_end_camera, gaussians, pipe, bg)["render"], skip_learn=iteration<opt.warmup)
                
            if event_cmap == "gray":
                event_loss = event_loss_call(
                    event_start_intensity, 
                    event_end_intensity, 
                    events[event_start_idx:event_end_idx],
                    select_coords, 
                    id_to_coords, 
                    "rgb",
                    device,
                    scene.image_h, 
                    scene.image_w
                ) * opt.event_loss_weight
            elif event_cmap == "color":
                event_loss = color_event_loss_call(
                    event_start_intensity, 
                    event_end_intensity, 
                    events[event_start_idx:event_end_idx],
                    select_coords, 
                    id_to_coords, 
                    "rgb",
                    device,
                    scene.image_h, 
                    scene.image_w,
                    id_to_color_map=id_to_color_map,
                    color_weight=torch.tensor([0.4, 0.2, 0.4]).to(device) if iteration > opt.warmup else None
                ) * opt.event_loss_weight
            else:
                raise Exception("Error: Unknown argument for edi.")
                
            loss += event_loss
            
            if edi_cmap == "gray":
                _, H, W = all_images_crf[0].shape
                
                all_images_crf_gray = []
                for i in all_images_crf:
                    all_images_crf_gray.append(
                        torch.mv(i.permute(1, 2, 0).reshape(-1, 3), rgbgray).reshape(H, W)
                    )
                gt_intensity = torch.mv(crf(gt_image, skip_learn=iteration<opt.warmup).permute(1, 2, 0).reshape(-1, 3), rgbgray).reshape(H, W)
                
                edi_window = events_edi_window[image_id]
                           
                edi_idx = random.randint(0, 8)
                edi_sharp_intensity = deblur_double_integral(gt_intensity, edi_window, idx=edi_idx, device=device)[None, :, :]
                pred_sharp_intensity = all_images_crf_gray[edi_idx][None, :, :]
                
                edi_Ll1 = l1_loss(pred_sharp_intensity, edi_sharp_intensity)
                edi_loss = (1.0 - opt.lambda_dssim) * edi_Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_sharp_intensity, edi_sharp_intensity))
                
                edi_window = events_edi_window[image_id]
                edi_idx = random.randint(0, 8)
                
                edi_sharp = deblur_double_integral(gt_image, edi_window, idx=edi_idx, device=device, color=True)
                pred_sharp = all_images[edi_idx]
                
                edi_color_Ll1 = l1_loss(pred_sharp, edi_sharp)
                edi_color_loss = (1.0 - opt.lambda_dssim) * edi_color_Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_sharp, edi_sharp))
            else:
                raise Exception("Error: Unknown argument for edi.")

            loss += edi_loss + edi_color_loss
            
            if edi_simul and iteration > opt.simul_start:
                all_images_crf_gray_mean = torch.mv(crf(image, skip_learn=iteration<opt.warmup).permute(1, 2, 0).reshape(-1, 3), rgbgray).reshape(H, W)
                
                edi_window = events_edi_window[image_id]           
                edi_idx = random.randint(0, 8)
                edi_sharp_intensity = deblur_double_integral(all_images_crf_gray_mean, edi_window, idx=edi_idx, device=device)[None, :, :]
                pred_sharp_intensity = all_images_crf_gray[edi_idx][None, :, :]
                
                edi_simul_Ll1 = l1_loss(pred_sharp_intensity, edi_sharp_intensity)
                edi_simul_loss = (1.0 - opt.lambda_dssim) * edi_simul_Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_sharp_intensity, edi_sharp_intensity))
                loss += edi_simul_loss

            if intensity and iteration > opt.warmup:
                all_images_crf_mean = torch.stack(all_images_crf, dim=0).mean(0)
                gt_crf = crf(gt_image, skip_learn=iteration<opt.warmup)

                intensity_Ll1 = l1_loss(all_images_crf_mean, gt_crf)
                intensity_loss = (1.0 - opt.lambda_dssim) * intensity_Ll1 + opt.lambda_dssim * (1.0 - ssim(all_images_crf_mean, gt_crf))
                loss += intensity_loss
        
        image_crop, org_image, mask = sample_patch(
            latent=image.unsqueeze(0).movedim(1, -1),
            image=gt_image.unsqueeze(0).movedim(1, -1),
            mask=None,
            patch_size=opt.rsd_patch_size,
            device=device
        )
            
        img_remap= F.interpolate(image_crop.movedim(-1,1), scale_factor=4, mode='bicubic', align_corners=False)
        img_remap = (img_remap * 2.0) - 1.0
        latent = guidance.encode_images(img_remap).movedim(1, -1)
        
        latents_noisy, pred_latents_noisy = guidance(latent, org_image, prompt_processor_output) #[NCHW]
        
        latents_noisy = latents_noisy.movedim(1,-1)
        pred_latents_noisy = pred_latents_noisy.movedim(1,-1)
        loss_rsd = F.l1_loss(latents_noisy, pred_latents_noisy, reduction="mean")
        loss += loss_rsd
        
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "t": {guidance.last_timestep}})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), event)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
s
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, event):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])     
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000, 140_000, 150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000, 140_000, 150_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000, 120_000, 130_000, 140_000, 150_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--event", type=bool, default = False)
    parser.add_argument("--event_cmap", type=str, default = 'gray')
    parser.add_argument("--edi_cmap", type=str, default = 'gray')
    parser.add_argument("--edi_simul", type=bool, default = False)
    parser.add_argument("--intensity", type=bool, default = False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.event,
        args.event_cmap,
        args.edi_cmap,
        args.edi_simul,
        args.intensity
    )

    # All done
    print("\nTraining complete.")
