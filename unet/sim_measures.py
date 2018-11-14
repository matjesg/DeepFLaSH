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
#  Compare masks using ROI-wise Jaccard Similarity
############################################################

def jaccard_roiwise(mask_a, mask_b, threshold=0.5, min_roi_pixel=15, roi_threshold=0.5):
    labels_a = utils.label_mask(mask_a, threshold=threshold, min_pixel=min_roi_pixel)
    labels_b = utils.label_mask(mask_b, threshold=threshold, min_pixel=min_roi_pixel)
    label_stack = np.dstack((labels_a, labels_b))

    comb_cadidates = np.unique(label_stack.reshape(-1, label_stack.shape[2]), axis=0)
    # Remove Zero Entries
    comb_cadidates = comb_cadidates[np.prod(comb_cadidates, axis=1) > 0]

    jac = [1 - jaccard((labels_a == x[0]).astype(np.uint8).flatten(), (labels_b == x[1]).astype(np.uint8).flatten()) for
           x in comb_cadidates]
    matches = np.sum(np.array(jac) >= roi_threshold)
    union = (np.unique((labels_a)).size-1) + (np.unique((labels_b)).size-1) - matches

    return(matches/union)