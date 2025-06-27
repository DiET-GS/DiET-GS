
import numpy as np
from utils.data import _is_pure_rotation_matrix, _get_slerp_interpolator, recenter_poses, spherify_poses
from itertools import product
import torch
import math

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


def events_pose_bspl(t, all_timestamps, all_poses):
    interpolator = _get_slerp_interpolator(all_timestamps, all_poses[:, :3, :3], all_poses[:, :3, 3])
    
    # Cannot interpolate beyond the available timestamps
    t = np.clip(t, a_min=all_timestamps.min(), a_max=all_timestamps.max())
    irots, itrans = interpolator(t)
    bottom = np.array([0, 0, 0, 1]).reshape(1, 1, -1).repeat(t.shape[0], axis=0)
    iposes = np.block([[irots, itrans[..., np.newaxis]], [bottom]])
    return iposes, None  # None here replaces the interpolated lookat targets, which we don't need


def interpolate_poses(t, all_timestamps, all_poses, bd_scale=1.0, recenter_partial=None, recenter=False):
        int_poses, _ = events_pose_bspl(t, all_timestamps, all_poses)

        int_poses = int_poses.astype(np.float32)

        int_poses[..., :3, 3] *= bd_scale

        if recenter:
            int_poses = recenter_poses(int_poses, c2w=recenter_partial)
            
        return int_poses


def interpolate_subpixel(x, y, v, w, h, image=None):
    image = image if image is not None else np.zeros((h, w), dtype=np.float32)

    if x.size == 0:
        return image

    # Implement the equation:
    # V(x,y) = \sum_i{ value * kb(x - xi) * kb(y - yi)}
    # We just consider the 4 integer coordinates around
    # each event coordinate, which will give a nonzero k_b()
    round_fns = (np.floor, np.ceil)

    k_b = lambda a: np.maximum(0, 1 - np.abs(a))
    xy_round_fns = product(round_fns, round_fns)
    for x_round, y_round in xy_round_fns:
        x_ref = x_round(x)
        y_ref = y_round(y)

        # Avoid summing the same contribution multiple times if the
        # pixel or time coordinate is already an integer. In that
        # case both floor and ceil provide the same ref. If it is an
        # integer, we only add it if the case #_round is torch.floor
        # We also remove any out of frame or bin coordinate due to ceil
        valid_ref = np.logical_and.reduce([
            np.logical_or(x_ref != x, x_round is np.floor),
            np.logical_or(y_ref != y, y_round is np.floor),
            x_ref < w, y_ref < h])
        x_ref = x_ref[valid_ref]
        y_ref = y_ref[valid_ref]

        if x_ref.shape[0] > 0:
            val = v[valid_ref] * k_b(x_ref - x[valid_ref]) * k_b(y_ref - y[valid_ref])
            np.add.at(image, (y_ref.astype(np.int64), x_ref.astype(np.int64)), val)

    return image


def brightness_increment_image(x, y, p, w, h, c_pos, c_neg, interpolate=True, threshold=False):
    assert c_pos is not None and c_neg is not None

    image_pos = np.zeros((h, w), dtype=np.float32)
    image_neg = np.zeros((h, w), dtype=np.float32)
    events_vals = np.ones([x.shape[0]], dtype=np.float32)

    pos_events = p > 0
    neg_events = np.logical_not(pos_events)

    if interpolate:
        image_pos = interpolate_subpixel(x[pos_events], y[pos_events], events_vals[pos_events], w, h, image_pos)
        image_neg = interpolate_subpixel(x[neg_events], y[neg_events], events_vals[neg_events], w, h, image_neg)
    else:
        np.add.at(image_pos, (y[pos_events].astype(np.int64), x[pos_events].astype(np.int64)), events_vals[pos_events])
        np.add.at(image_neg, (y[neg_events].astype(np.int64), x[neg_events].astype(np.int64)), events_vals[neg_events])
    
    if not threshold:        
        image = image_pos.astype(np.float32) - image_neg.astype(np.float32)
    else:
        image = image_pos.astype(np.float32) * c_pos - image_neg.astype(np.float32) * c_neg
    return image


def inner_double_integral(bii, device):
    assert bii.shape[0] % 2 == 0
    N = bii.shape[0] // 2

    images = []
    # Left part of the interval from f-T/2 to f
    for i in range(N):
        images.append(- bii[i:N].sum(axis=0))
    # Frame at f
    images.append(torch.zeros_like(images[0]).to(device))
    # Right part of the interval from f to f+T/2
    for i in range(N):
        images.append(+ bii[N:N + 1 + i].sum(axis=0))

    images = torch.stack(images, axis=0)
    return images


def deblur_double_integral(blurry, bii, idx=0, device=None, color=False):
    N = bii.shape[0] // 2
     
    if color:
        bii = bii[:, None, :, :].repeat(1, 3, 1, 1)
    images = inner_double_integral(bii, device)
    
    if idx == 4:
        sharp = ((2*N+1) * blurry / torch.exp(images).sum(axis=0))
    elif idx < 4:
        sharp = ((2*N+1) * blurry / torch.exp(images).sum(axis=0)) / torch.exp(bii[idx:4].sum(axis=0))
    else:
        sharp = ((2*N+1) * blurry / torch.exp(images).sum(axis=0)) * torch.exp(bii[4:idx+1].sum(axis=0))

    return sharp

def deblur_double_integral_continuous(blurry, idi, events, id_to_coords, t=0, mt=0, device=None):
    N = events[:, 1][-1] - events[:, 1][0] + 1
    mid_timestamp = mt
    
    if t == mid_timestamp:
        sharp = (N * blurry / idi)
    elif t < mid_timestamp:
        idx_events_left = torch.searchsorted(events[:, 1], torch.tensor([t]).to(device))
        idx_events_right = torch.searchsorted(events[:, 1], torch.tensor([mid_timestamp]).to(device), side="right")
        
        ev = events[idx_events_left:idx_events_right]
        x, y = id_to_coords[ev[:, 0].long()].T.cpu().numpy()
        p = ev[:, 2].cpu().numpy()

        bii = brightness_increment_image(x, y, p, 346, 260, 0.25, 0.25, interpolate=True)  # [H, W] -> 346, 260
        bii = torch.from_numpy(bii).to(device)
        
        sharp = (N * blurry / idi) / torch.exp(bii)
    elif t > mid_timestamp:
        idx_events_left = torch.searchsorted(events[:, 1], torch.tensor([mid_timestamp]).to(device))
        idx_events_right = torch.searchsorted(events[:, 1], torch.tensor([t]).to(device), side="right")
        
        ev = events[idx_events_left:idx_events_right]
        x, y = id_to_coords[ev[:, 0].long()].T.cpu().numpy()
        p = ev[:, 2].cpu().numpy()

        bii = brightness_increment_image(x, y, p, 346, 260, 0.25, 0.25, interpolate=True)  # [H, W] -> 346, 260
        bii = torch.from_numpy(bii).to(device)
        
        sharp = (N * blurry / idi) * torch.exp(bii)
        
    return sharp


def color_event_map_func(image, id_to_color_map, device):
    rgb2grey = torch.tensor([0.299,0.587,0.114]).to(device)
    C, H, W = image.shape
    image = image.permute(1, 2, 0).reshape(-1, C)
    id_to_color_map = id_to_color_map.reshape(-1, C)
    
    image_gray = torch.mv(image, rgb2grey)
    
    color_mask = id_to_color_map.sum(dim=1) == 0

    id_to_color_map_ = id_to_color_map.clone()
    id_to_color_map_[color_mask, :] = torch.tensor([True, False, False]).to(device)
    
    processed_image = image[id_to_color_map_]
    processed_image[color_mask] = image_gray[color_mask]
    
    return processed_image.reshape(H, W), color_mask


def polarity_func(timestamp, events, id_to_coords, device):
    if timestamp < events[:, 1][0] + 1000:
        start_timestamp, end_timestamp = events[:, 1][0], events[:, 1][0] + 1000
    elif timestamp > events[:, 1][-1] - 1000:
        start_timestamp, end_timestamp = events[:, 1][-1] - 1000, events[:, 1][-1]
    else:
        start_timestamp, end_timestamp = timestamp - 1000, timestamp + 1000

    event_start_idx = torch.searchsorted(events[:, 1], torch.tensor([start_timestamp]).to(device))
    event_end_idx = torch.searchsorted(events[:, 1], torch.tensor([end_timestamp]).to(device), side="right")

    event_data = events[event_start_idx:event_end_idx]

    x, y = id_to_coords[event_data[:, 0].long()].T.cpu().numpy()
    p = event_data[:, 2].cpu().numpy()

    bii = brightness_increment_image(x, y, p, 346, 260, 0.25, 0.25, interpolate=True)

    polarity_pos = np.zeros(bii.shape)
    polarity_neg = np.zeros(bii.shape)

    polarity_pos[bii > 0] = bii[bii > 0]
    polarity_neg[bii < 0] = bii[bii < 0]
    
    polarity = np.stack([polarity_pos, polarity_neg], axis=-1)

    return torch.from_numpy(polarity).float().to(device)


def events_cal_func(all_images_crf_gray, edi_window, device):
    pseudo_edi_window_list = []
    for i in range(len(all_images_crf_gray) - 1):
        intensity_left = all_images_crf_gray[i]
        intensity_right = all_images_crf_gray[i+1]
        
        pseudo_edi_window = torch.zeros_like(intensity_left)
        
        pos_idx = edi_window[i] > 0
        neg_idx = edi_window[i] < 0
        
        pseudo_edi_window_full = (lin_log(intensity_right*255) - lin_log(intensity_left*255))
        pseudo_edi_window[pos_idx] = pseudo_edi_window_full[pos_idx]
        pseudo_edi_window[neg_idx] = pseudo_edi_window_full[neg_idx]
        
        pseudo_edi_window_list.append(pseudo_edi_window)
    pseudo_edi_window_list = torch.stack(pseudo_edi_window_list, dim=0)
    return pseudo_edi_window_list.to(device)