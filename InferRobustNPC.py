from glob import glob
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List
import torch
import torch.nn as nn
from generic_UNet import InitWeights_He
import pickle
import os
import torch.nn.functional as F
from generic_UNet import Generic_UNet
from tqdm import tqdm
prefix = "version5"

modelfile = "./pretrained_model/robustnpc.model"

print("Inference")
resolution_index = 1
num_classes = 1
base_num_features = 32 
patch_size = [48, 192, 192]
# patch_size = [32, 224, 256] # in the original paper is 48, 192, 192, you can set it to be a larger value for less inference time with similar performance if your GPU memory is allowed.
pool_op_kernel_sizes = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 2]]
conv_kernel_sizes = [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
current_spacing = [3.0, 0.46880001, 0.46880001]

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

net = Generic_UNet(1, base_num_features, num_classes + 1, len(pool_op_kernel_sizes), 2, 2,
                   nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d,
                   dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
                   InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes, False, True, True)

net.cuda()
checkpoint = torch.load(modelfile)
weights = checkpoint['state_dict']
net.load_state_dict(weights, strict=False)
net.eval()
net.half()


# summary(net, (1, 80, 160, 192), batch_size=1)


def _get_arr(path):
    sitkimg = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitkimg)
    return arr, sitkimg


def _write_arr(arr, path, info=None):
    sitkimg = sitk.GetImageFromArray(arr)
    if info is not None:
        sitkimg.CopyInformation(info)
    sitk.WriteImage(sitkimg, path)


def get_do_separate_z(spacing, anisotropy_threshold=2):
    # do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    do_separate_z = spacing[-1] > anisotropy_threshold
    return do_separate_z


def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                      image_size: Tuple[int, ...],
                                      step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map


gaussian_mask = torch.from_numpy(_get_gaussian(patch_size)[np.newaxis, np.newaxis]).cuda().half().clamp_min_(1e-4)


def predict(arr):
    prob_map = torch.zeros((1, num_classes + 1,) + arr.shape).half().cuda()
    # arr_clip = np.clip(arr, clip_min, clip_max)
    # raw_norm = (arr_clip - mean) / std
    raw_norm = (arr - arr.mean()) / arr.std()
    steps = _compute_steps_for_sliding_window(patch_size, raw_norm.shape, 0.7)
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
    if 1:
        print("data shape:", raw_norm.shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", steps)
        print("number of tiles:", num_tiles)

    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]
                with torch.no_grad():
                    tensor_arr = torch.from_numpy(
                        raw_norm[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis, np.newaxis]).cuda().half()
                    seg_pro = net(tensor_arr)
                    _pred = seg_pro * gaussian_mask
                    prob_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred
    torch.cuda.empty_cache()
    return prob_map.detach().cpu()


def itk_change_spacing(src_itk, output_spacing, interpolate_method='Linear'):
    assert interpolate_method in ['Linear', 'NearestNeighbor']
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()

    re_sample_scale = tuple(np.array(src_spacing) / np.array(output_spacing).astype(np.float))
    re_sample_size = tuple(np.array(src_size).astype(np.float) * np.array(re_sample_scale))

    re_sample_size = [int(round(x)) for x in re_sample_size]
    output_spacing = tuple((np.array(src_size) / np.array(re_sample_size)) * np.array(src_spacing))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetOutputPixelType(src_itk.GetPixelID())
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(re_sample_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    return re_sampler.Execute(src_itk)


def resample_image_to_ref(image, ref, interp=sitk.sitkNearestNeighbor, pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(interp)
    return resample.Execute(image)


def Inference3D(rawf, save_path=None):
    arr_raw, sitk_raw = _get_arr(rawf)
    origin_spacing = sitk_raw.GetSpacing()
    rai_size = sitk_raw.GetSize()

    if get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1]):
        print('preprocessing: do seperate z.....')
        img_arr = []
        for i in range(rai_size[-1]):
            img_arr.append(sitk.GetArrayFromImage(itk_change_spacing(sitk_raw[:, :, i], current_spacing[::-1][:-1])))
        img_arr = np.array(img_arr)
        img_sitk = sitk.GetImageFromArray(img_arr)
        img_sitk.SetOrigin(sitk_raw.GetOrigin())
        img_sitk.SetDirection(sitk_raw.GetDirection())
        img_sitk.SetSpacing(tuple(current_spacing[::-1][:-1]) + (origin_spacing[-1],))
        img_arr = sitk.GetArrayFromImage(
            itk_change_spacing(img_sitk, current_spacing[::-1], interpolate_method='NearestNeighbor'))
    else:
        img_arr = sitk.GetArrayFromImage(itk_change_spacing(sitk_raw, current_spacing[::-1]))
    pad_flag = 0
    padzyx = np.clip(np.array(patch_size) - np.array(img_arr.shape), 0, 1000)
    if np.any(padzyx > 0):
        pad_flag = 1
        pad_left = padzyx // 2
        pad_right = padzyx - padzyx // 2
        img_arr = np.pad(img_arr,
                         ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))

    prob_map = predict(img_arr)

    if pad_flag:
        prob_map = prob_map[:, :,
                   pad_left[0]: img_arr.shape[0] - pad_right[0],
                   pad_left[1]: img_arr.shape[1] - pad_right[1],
                   pad_left[2]: img_arr.shape[2] - pad_right[2]]
    del img_arr

    '''
    FYI, In order to smooth the organ edge, is there any better choice ?
    '''

    if get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1]):
        print('postpreprocessing: do seperate z......')
        prob_map_interp_xy = torch.zeros(
            list(prob_map.size()[:2]) + [prob_map.size()[2], ] + list(sitk_raw.GetSize()[::-1][1:]), dtype=torch.half)

        for i in range(prob_map.size(2)):
            prob_map_interp_xy[:, :, i] = F.interpolate(prob_map[:, :, i].cuda().float(),
                                                        size=sitk_raw.GetSize()[::-1][1:],
                                                        mode="bilinear").detach().half().cpu()
        del prob_map

        prob_map_interp = np.zeros(list(prob_map_interp_xy.size()[:2]) + list(sitk_raw.GetSize()[::-1]),
                                   dtype=np.float16)

        for i in range(prob_map_interp.shape[1]):
            prob_map_interp[:, i] = F.interpolate(prob_map_interp_xy[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="nearest").detach().half().cpu().numpy()
        del prob_map_interp_xy

    else:
        prob_map_interp = np.zeros(list(prob_map.size()[:2]) + list(sitk_raw.GetSize()[::-1]), dtype=np.float16)

        for i in range(prob_map.size(1)):
            prob_map_interp[:, i] = F.interpolate(prob_map[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="trilinear").detach().half().cpu().numpy()
        del prob_map

    segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)
    del prob_map_interp
    pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    pred_sitk.CopyInformation(sitk_raw)
    pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)
    if save_path is None:
        save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
    else:
        uid = rawf.split("/")[-1].replace(".nii.gz", "_pred.nii.gz")
        save_dir = os.path.join(save_path, uid)
    sitk.WriteImage(pred_sitk, save_dir)


rawf = "example.nii.gz"
Inference3D(rawf)

