import torch
from torch import nn
from torch.nn import functional as F
import torch.optim
import matplotlib
import os
import numpy as np
from dataset.data_split import split_data,Concat
from scipy import ndimage
from skimage.measure import label, regionprops
import nibabel as nib
from skimage.morphology import disk, remove_small_objects


class MixLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x, y):
        lf, lfw = [], []
        for i, v in enumerate(self.args):
            if i % 2 == 0:
                lf.append(v)
            else:
                lfw.append(v)
        mx = sum([w*l(x,y) for l, w in zip(lf, lfw)])
        return mx

class DiceLoss(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image
    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x*y, x+y]]
        dc = (2*i+1)/(u+1)
        dc = 1-dc.mean()
        return dc


def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :] + torch.sum(targets[:, class_index, :, :, :]))
    dice = (2. * inter + 1) / (union + 1)
    return dice


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.contiguous().view(num, -1)
        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


def _remove_low_probs(pred, prob_thresh):
    pred = np.where(pred > prob_thresh, pred, 0)
    return pred


def _remove_spine_fp(pred, image, bone_thresh):
    image_bone = image > bone_thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0
    
    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)
    return pred


def _post_process(pred, image, prob_thresh, bone_thresh, size_thresh):

    # remove connected regions with low confidence
    pred = _remove_low_probs(pred, prob_thresh)

    # remove spine false positives
    pred = _remove_spine_fp(pred, image, bone_thresh)

    # remove small connected regions
    pred = _remove_small_objects(pred, size_thresh)

    return pred


def _predict_single_image(model, dataloader, postprocess, prob_thresh,
        bone_thresh, size_thresh):
    pred = np.zeros(dataloader.dataset.image.shape)#Output_shape
    crop_size = dataloader.dataset.crop_size
    #if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()
    with torch.no_grad():
        for _,sample in enumerate(dataloader):
            images, centers = sample
            images = images.cuda()
            output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)
            for i in range(len(centers)):
                center_x, center_y, center_z = centers[i]
                cur_pred_patch = pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ]
                pred[
                    center_x - crop_size // 2:center_x + crop_size // 2,
                    center_y - crop_size // 2:center_y + crop_size // 2,
                    center_z - crop_size // 2:center_z + crop_size // 2
                ] = np.where(cur_pred_patch > 0, np.mean((output[i], cur_pred_patch), axis=0), output[i])
    if postprocess:
        pred = _post_process(pred, dataloader.dataset.image, prob_thresh,
            bone_thresh, size_thresh)
    return pred

def _make_submission_files(pred, affine):
    pred_label = label(pred > 0).astype(np.int16)
    # 这里阈值作为参数调整 0.9的时候froc 为 0.21 0.8为0.25 0.7 为0.28 0.6为 0.29  0.5为 0.32
                                                # 0.4 为  0.37 0.3 0.43 0.2 0.4625 0.1 0.5336
    pred_label[pred_label>0.5]=1
    pred_image = nib.Nifti1Image(pred_label, affine)
    return pred_image
# def _make_submission_files(pred, image_id, affine):
#     pred_label = label(pred > 0.8).astype(np.int16)
#     pred_regions = regionprops(pred_label, pred)
#     pred_index = [0] + [region.label for region in pred_regions]
#     pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
#     # placeholder for label class since classifaction isn't included
#     pred_label_code = [0] + [1] * int(pred_label.max())
#     pred_label [pred_label > 0.5] = 1
#     pred_image = nib.Nifti1Image(pred_label, affine)
#     pred_info = pd.DataFrame({
#         "public_id": [image_id] * len(pred_index),
#         "label_id": pred_index,
#         "confidence": pred_proba,
#         "label_code": pred_label_code
#     })

#     return pred_image, pred_info
