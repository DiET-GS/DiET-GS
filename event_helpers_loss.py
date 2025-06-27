import torch
import math
import random
import numpy as np
from utils.event_utils import *


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


def event_loss_call(
    rgb_start, 
    rgb_end, 
    event_data, 
    select_coords, 
    id_to_coords, 
    rgb2gray, 
    device, 
    resolution_h=None, 
    resolution_w=None,
    ):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    if rgb2gray == "rgb":
        rgb2grey = torch.tensor([0.299,0.587,0.114]).to(device)
    elif rgb2gray == "ave":
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to(device)

    rgb_start = rgb_start.permute(1, 2, 0)[select_coords[:, 0], select_coords[:, 1]]
    rgb_end = rgb_end.permute(1, 2, 0)[select_coords[:, 0], select_coords[:, 1]]

    thres = (lin_log(torch.mv(rgb_end, rgb2grey)*255) - lin_log(torch.mv(rgb_start, rgb2grey)*255)) / 0.2 # FIXME:

    x, y = id_to_coords[event_data[:, 0].long()].T.cpu().numpy()
    p = event_data[:, 2].cpu().numpy()

    bii = brightness_increment_image(x, y, p, resolution_w, resolution_h, 0.2, 0.2, interpolate=True) # FIXME:

    bii = bii[select_coords[:, 0], select_coords[:, 1]]
    bii = torch.from_numpy(bii).to(device)

    pos = bii > 0
    neg = bii < 0

    loss_pos = torch.mean(((thres * pos) - (bii * pos)) ** 2)
    loss_neg = torch.mean(((thres * neg) - (bii * neg)) ** 2)
    
    event_loss = torch.mean(loss_pos + loss_neg)
    return event_loss


def color_event_loss_call(
    rgb_start, 
    rgb_end, 
    event_data, 
    select_coords, 
    id_to_coords, 
    rgb2gray, 
    device, 
    resolution_h=None, 
    resolution_w=None,
    id_to_color_map=None,
    color_weight=None,
    ):
    '''
    simulate the generation of event stream and calculate the event loss
    '''
    if rgb2gray == "rgb":
        rgb2grey = torch.tensor([0.299,0.587,0.114]).to(device)
    elif rgb2gray == "ave":
        rgb2grey = torch.tensor([1/3, 1/3, 1/3]).to(device)
    
    rgb_start = rgb_start.permute(1, 2, 0)[select_coords[:, 0], select_coords[:, 1]]
    rgb_end = rgb_end.permute(1, 2, 0)[select_coords[:, 0], select_coords[:, 1]]
    id_to_color_map = id_to_color_map[select_coords[:, 0], select_coords[:, 1]]

    thres = (lin_log(rgb_end * 255) - lin_log(rgb_start * 255)) / 0.25
    thres = thres[id_to_color_map]
    
    color_mask = id_to_color_map.sum(dim=1) != 0
    
    x, y = id_to_coords[event_data[:, 0].long()].T.cpu().numpy()
    p = event_data[:, 2].cpu().numpy()

    bii = brightness_increment_image(x, y, p, resolution_w, resolution_h, 0.25, 0.25, interpolate=True)

    bii = bii[select_coords[:, 0], select_coords[:, 1]]
    bii = torch.from_numpy(bii).to(device)
    bii = bii[color_mask]
    
    if color_weight is not None:
        id_to_color_map = id_to_color_map[color_mask]
        color_idx = torch.where(id_to_color_map)[1]
        color_weight = color_weight[color_idx]
    else:
        color_weight = torch.ones(bii.shape[0]).to(device)
    
    pos = bii > 0
    neg = bii < 0

    loss_pos = torch.mean((((thres * pos - bii * pos) ** 2) * color_weight).sum() / color_weight.sum())
    loss_neg = torch.mean((((thres * neg - bii * neg) ** 2) * color_weight).sum() / color_weight.sum())
    
    event_loss = torch.mean(loss_pos + loss_neg)
    return event_loss

