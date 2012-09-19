from _imseg_rand_forest import RandomForestClassifier
import _imseg_rand_forest as rand_forest
from _imseg_rand_forest_image_features import TextonFeatureFactory, TextonFeature, TextonPredict, IntegralFeatureFactory, IntegralFeature, IntegralPredict
from super_pixels import SuperPixels

import numpy as np
import cv2


def convert_labels_to_integrals(label_mask, num_vals):
    """
    Args:
        label_mask: 
        num_vals: 
    """
    masks = []
    print(label_mask.shape)
    for x in range(num_vals):
        m = np.asfarray(label_mask == x)
        m = cv2.integral(m)
        masks.append(m)
    out = np.dstack(masks)
    return np.ascontiguousarray(out)


def convert_labels_probs_to_integrals(label_mask, max_probs, num_vals):
    """
    Args:
        label_mask:
        max_probs:
        num_vals: 
    """
    masks = []
    for x in range(num_vals):
        m = np.asfarray(label_mask == x) * max_probs
        m = cv2.integral(m)
        masks.append(m)
    return np.ascontiguousarray(np.dstack(masks))


def convert_all_probs_to_integrals(all_probs):
    """
    Args:
        all_probs: 
    """
    masks = []
    for x in range(all_probs.shape[2]):
        m = np.ascontiguousarray(all_probs[:, :, x], dtype=np.float64)
        masks.append(cv2.integral(m))
    return np.ascontiguousarray(np.dstack(masks))


def make_color_key(colors, class_names, el=32, width=500, pos_frac=.75):
    color_key = np.zeros((len(colors) * el, width, 3), np.uint8)
    for num, (color, class_name) in enumerate(zip(colors, class_names)):
        color = color[::-1]
        if np.mean(color) < 50:
            text_color = (255, 255, 255)
        else:
            text_color = (0, 0, 0)
        cv2.fillPoly(color_key, np.array([[(0, (num * el)), (0, (num + 1) * el), (width, (num + 1) * el), (width, num * el)]]), color.tolist())
        cv2.putText(color_key, class_name, (0, (num * el) + int(el * pos_frac)), 2, cv2.FONT_HERSHEY_PLAIN, text_color)
    return color_key

