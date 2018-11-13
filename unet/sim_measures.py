"""
Sim_measures
Similarity functions to compare segmentation maps (masks)
Licensed under the MIT License (see LICENSE for details)
Written by Matthias Griebel
"""

from scipy.spatial.distance import jaccard
from unet import utils
import numpy as np

############################################################
#  Compare masks using pixelwise Jaccard Similarity
############################################################

def jaccard_pixelwise(mask_a, mask_b, threshold=0.5):
    mask_a = (mask_a > threshold).astype(np.uint8)
    mask_b = (mask_b > threshold).astype(np.uint8)
    jac_dist = jaccard(mask_a.flatten(), mask_b.flatten())

    return (1 - jac_dist)

############################################################
#  Calculate Jaccard Similarity between two ROIs
# (using coordinates)
############################################################

def jaccard_roi(a,b):
  x = [':'.join(x) for x in a.astype(str).tolist()]
  y = [':'.join(x) for x in b.astype(str).tolist()]
  z = np.unique(np.concatenate((x,y)))

  return((len(x) + len(y) - z.size)/z.size)

############################################################
#  Pair generator function for ROI matching
############################################################

def match_pairs(regions_a, regions_b, roi_threshold=0.5):
    match_a = np.array([])
    match_b = np.array([])

    for a in regions_a:
        for b in regions_b:
            if not (np.isin(a.label, match_a) or np.isin(b.label, match_b)):
                jacc = jaccard_roi(a.coords, b.coords)

                if jacc >= roi_threshold:
                    match_a = np.append(match_a, a.label)
                    match_b = np.append(match_b, b.label)

                    yield a.label, b.label, jacc

############################################################
#  Compare masks using ROI-wise Jaccard Similarity
############################################################

def jaccard_roiwise(mask_a, mask_b, threshold=0.5, min_roi_pixel=15, roi_threshold=0.5):
    regions_a = utils.roi_eval(mask_a, threshold=threshold, min_pixel=min_roi_pixel)
    regions_b = utils.roi_eval(mask_b, threshold=threshold, min_pixel=min_roi_pixel)
    res_list = list(match_pairs(regions_a, regions_b, roi_threshold=roi_threshold))

    return (len(res_list) / (len(regions_a) + len(regions_b) - len(res_list)))