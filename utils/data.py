import os
import numpy as np

import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

import random


def _is_pure_rotation_matrix(M):
    """
    Check if a given matrix is a pure rotation matrix.
    :param M: a numpy ndarray of shape (N, 3, 3)
    :return: a boolean ndarray of shape (N,) indicating whether each matrix is a pure rotation matrix
    """
    # Check if each matrix in the list is square
    if M.shape[1] != M.shape[2]:
        return False

    # Check if the determinant of each matrix is +1
    det = np.linalg.det(M)
    is_det_close_to_one = np.isclose(det, 1.0)
    if not np.all(is_det_close_to_one):
        return False

    # Check if the transpose of each matrix is its inverse
    MT = np.transpose(M, (0, 2, 1))
    M_inv = np.linalg.inv(M)
    is_MT_close_to_M_inv = np.allclose(MT, M_inv, atol=5e-7)
    if not np.all(is_MT_close_to_M_inv):
        return False
    return True


def _get_slerp_interpolator(tss_poses_us, poses_rots, poses_trans):
    """
    Input
    :tss_poses_ns list of known tss
    :poses_rots list of 3x3 np.arrays
    :poses_trans list of 3x1 np.arrays
    :tss_query_ns list of query tss

    Returns:
    :rots list of rots at tss_query_ns
    :trans list of translations at tss_query_ns
    """
    # Setup Rot interpolator
    rot_interpolator = Slerp(tss_poses_us, R.from_matrix(poses_rots))
    # Setup trans interpolator
    trans_interpolator = interp1d(x=tss_poses_us, y=poses_trans, axis=0, kind="cubic", bounds_error=True)

    # Create interpolator as a closure to avoid creating the
    # trans_interpolator and rot_interpolator each time
    def interpolator(tss_query_ns):        
        tss_query_ns = np.clip(tss_query_ns, tss_poses_us[0], tss_poses_us[-1])
        # Query rot interpolator
        rots = rot_interpolator(tss_query_ns).as_matrix()
        # Query trans interpolator
        trans = trans_interpolator(tss_query_ns)

        return rots, trans
    return interpolator


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # view direction
        # c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        # camera poses
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def render_path_epi(c2w, up, rads, N):
    render_poses = []
    hwf = c2w[:, 4:5]

    for theta in np.linspace(-1, 1, N + 1)[:-1]:
        # view direction
        c = np.dot(c2w[:3, :4], np.array([theta, 0, 0, 1.]) * rads)
        # camera poses
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, 1, 0.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses, c2w=None, return_c2w=False):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    if c2w is None:
        c2w = poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_

    if return_c2w:
        return poses, c2w
    else:
        return poses


#####################


def spherify_poses(poses, bds, state=None, return_state=False):
    c2w, up, sc, radcircle, zh = state if state is not None else [None] * 5
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    if state is None:
        def min_line_dist(rays_o, rays_d):
            A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -A_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)

        vec0 = normalize(up)
        vec1 = normalize(np.cross([.1, .2, .3], vec0))
        vec2 = normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    if state is None:
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc

        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
    else:
        rad = 1. / sc
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc

    new_poses = []
    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    if return_state:
        state = [c2w, up, sc, radcircle, zh]
        return poses_reset, new_poses, bds, state
    return poses_reset, new_poses, bds


def sample_patch(latent, image, mask, patch_size, device, apply_mask=False):
    B,H,W,C = latent.shape

    ph=pw=patch_size

    # get masked crop coords
    coord_y, coord_x = torch.meshgrid(torch.arange(H,device=device), torch.arange(W,device=device), indexing='ij')
    coord_x = coord_x.flatten()
    coord_y = coord_y.flatten()
    x_mask = torch.all(torch.stack([coord_x>(pw//2), coord_x<(W-pw//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
    y_mask = torch.all(torch.stack([coord_y>(ph//2), coord_y<(H-ph//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW

    if mask is not None:
        fg_mask = mask.reshape(B,-1) # B,HW
        comp_masks = torch.all(torch.stack([x_mask,y_mask,fg_mask]),dim=0)
        fg_masks = fg_mask.view(B,H,W,1)
    else:
        comp_masks = torch.all(torch.stack([x_mask,y_mask]),dim=0)

    rays = image.view(B,H,W,3)
    latent = latent.view(B,H,W,C)
    
    patch_rays_list = []
    patch_latent_list = []
    patch_mask_list = []

    for i, comp_mask in enumerate(comp_masks):
        valid_coord_x = coord_x[comp_mask]
        valid_coord_y = coord_y[comp_mask]
        assert len(valid_coord_x) == len(valid_coord_y)
        sample_idx = random.randint(0,len(valid_coord_x)-1)
        h_sample = valid_coord_y[sample_idx]
        w_sample = valid_coord_x[sample_idx]

        # edge sampling for LLFF dataset
        if not apply_mask:
            h_sample = random.randint(0,H)
            w_sample = random.randint(0,W)
            if h_sample < (ph//2):
                h_sample = ph//2
            if w_sample < (pw//2):
                w_sample = pw//2
            if h_sample > (H-ph//2):
                h_sample = (H-ph//2)
            if w_sample > (W-pw//2):
                w_sample = (W-pw//2)

        center = (h_sample,w_sample)
        crop_size = (ph,pw)
        crop_box = (int(center[1]-crop_size[1]/2), int(center[0]-crop_size[0]/2), int(center[1]+crop_size[1]/2), int(center[0]+crop_size[0]/2)) #left,up,right,down
        
        patch_rays = rays[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,3) #(128*128,3)
        patch_latent = latent[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,C) #(128*128,1)
   
        patch_rays_list.append(patch_rays)
        patch_latent_list.append(patch_latent)

        if apply_mask:
            patch_mask = fg_masks[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,1) #(128*128,1)
            patch_mask_list.append(patch_mask)
        
    patch_rays = torch.stack(patch_rays_list)
    patch_latent = torch.stack(patch_latent_list)

    patch_rays = patch_rays.reshape(B,ph,pw,3)
    patch_latent = patch_latent.reshape(B,ph,pw,C)

    if apply_mask:
        patch_mask = torch.stack(patch_mask_list)
        patch_mask = patch_mask.reshape(B,ph,pw,1)
    else:
        patch_mask = None

    return patch_latent, patch_rays, patch_mask


def sample_patch_broad(latent, image, mask, patch_size, device, apply_mask=False):
        B,H,W,C = latent.shape

        ph=pw=patch_size

        # get masked crop coords
        coord_y, coord_x = torch.meshgrid(torch.arange(H,device=device), torch.arange(W,device=device), indexing='ij')
        coord_x = coord_x.flatten()
        coord_y = coord_y.flatten()
        x_mask = torch.all(torch.stack([coord_x>(pw//2), coord_x<(W-pw//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        y_mask = torch.all(torch.stack([coord_y>(ph//2), coord_y<(H-ph//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        
        if mask is not None:
            fg_mask = mask.reshape(B,-1) # B,HW
            comp_masks = torch.all(torch.stack([x_mask,y_mask,fg_mask]),dim=0)
            fg_masks = fg_mask.view(B,H,W,1)
        else:
            comp_masks = torch.all(torch.stack([x_mask,y_mask]),dim=0)
            
        valid_coord_x = coord_x[comp_masks[0]]
        valid_coord_y = coord_y[comp_masks[0]]
        assert len(valid_coord_x) == len(valid_coord_y)
        sample_idx = random.randint(0,len(valid_coord_x)-1)
        h_sample = valid_coord_y[sample_idx]
        w_sample = valid_coord_x[sample_idx]

        # edge sampling for LLFF dataset
        if not apply_mask:
            h_sample = random.randint(0,H)
            w_sample = random.randint(0,W)
            if h_sample < (ph//2):
                h_sample = ph//2
            if w_sample < (pw//2):
                w_sample = pw//2
            if h_sample > (H-ph//2):
                h_sample = (H-ph//2)
            if w_sample > (W-pw//2):
                w_sample = (W-pw//2)

        center = (h_sample,w_sample)
        crop_size = (ph,pw)
        crop_box = (int(center[1]-crop_size[1]/2), int(center[0]-crop_size[0]/2), int(center[1]+crop_size[1]/2), int(center[0]+crop_size[0]/2))
            
        rays = image.view(B,H,W,3)
        latent = latent.view(B,H,W,C)
    
        patch_rays_list = []
        patch_latent_list = []
        patch_mask_list = []

        for i in range(len(comp_masks)):
            patch_rays = rays[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,3) #(128*128,3)
            patch_latent = latent[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,C) #(128*128,1)
   

            patch_rays_list.append(patch_rays)
            patch_latent_list.append(patch_latent)

            if apply_mask:
                patch_mask = fg_masks[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,1) #(128*128,1)
                patch_mask_list.append(patch_mask)
        
        patch_rays = torch.stack(patch_rays_list)
        patch_latent = torch.stack(patch_latent_list)

        patch_rays = patch_rays.reshape(B,ph,pw,3)
        patch_latent = patch_latent.reshape(B,ph,pw,C)

        if apply_mask:
            patch_mask = torch.stack(patch_mask_list)
            patch_mask = patch_mask.reshape(B,ph,pw,1)
        else:
            patch_mask = None

        return patch_latent, patch_rays, patch_mask, crop_box