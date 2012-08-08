#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "forest_image_features_aux.h"


#define BOUNDS(i, j, height, width) (0 <= (i) && (i) < height && 0 <= (j) && (j) < width)


inline int data_samp(uint8_t *data, int i, int j, int b, int height, int width) {
    if (0 > b || b >= 7)
        return 0;
    if (i < 0)
        i = 0;
    else if (i >= height)
        i = height - 1;
    if (j < 0)
        j = 0;
    else if (j >= width)
        j = width - 1;
    return data[(i * width + j) * 7 + b];
}



inline double integral_samp(double *data, int i, int j, int mask_num, int height, int width, int num_masks) {
    if (0 <= i && i < height && 0 <= j && j < width && 0 <= mask_num && mask_num < num_masks)
        return data[(i * width + j) * num_masks + mask_num];
    return 0.;
}

inline void sum_thresh_array_fast(uint8_t *thresh_mask, int num_threshs, int num_points, int32_t *out_false, int32_t *out_true) {
    /* Take in a boolean array (made for feature_func_many) and find the number of True/False values in each row (output into out_false, out_true)

       Args:
           thresh_mask: (num_thresh, num_points)
           num_threshs:
           num_points:
           out_false: Sum of False values in each column (num_thresh)
           out_true: Sum of True values in each column (num_thresh)
     */
    int i, j;
    int32_t *outs[2];
    outs[0] = out_false;
    outs[1] = out_true;
    for (i = 0; i < num_threshs; ++i, ++outs[0], ++outs[1], thresh_mask += num_points)
        for (j = 0; j < num_points; ++j)
            ++(*outs[thresh_mask[j]]);
}


void predict_multi_tree_max_argmax(int size, int num_classes, int num_trees, int all_probs, int32_t *out_leaves, double *leaves,
                                   double *out_probs, int32_t *leaf_class, double *leaf_prob) {
    /* Computes the argmax and max over out_probs.  If all_probs is true, then normalize out_probs.  This is in C because it was a major bottleneck.

       Args:
           size: Height * width
           num_classes: Number of classes
           num_trees: Number of trees (leaf_prob and potentially out_probs are normalized by this)
           all_probs: If true then normalize out_probs.
           out_leaves: Provided leaf assignments  (from classifier) with size of (num_trees, size)
           leaves: Provided leaf probabilities with size of (leaves, num_classes)

      Returns:
           out_probs: Its size must be (size, num_classes), and filled with the sums of probabilities for each tree.  If all_probs is True, then this will be normalized.
           leaf_class: Will be populated with the argmax over the classes, its size must be "size".
           leaf_prob: Will be populated with the max over the classes normalized, its size must be "size" and initialized to 0.
     */
    int i, j, k;
    int total_el = size * num_classes;
    double inv_norm = 1. / num_trees;
    double *cur_out_probs;
    int32_t *cur_out_leaves = out_leaves;
    double *cur_leaf;
    for (k = 0; k < num_trees; ++k, cur_out_leaves += size)
        for (i = 0, cur_out_probs = out_probs; i < size; ++i, cur_out_probs += num_classes) {
            cur_leaf = leaves + num_classes * cur_out_leaves[i];
            for (j = 0; j < num_classes; ++j)
                cur_out_probs[j] += cur_leaf[j];
        }
    if (all_probs)
        for (i = 0; i < total_el; ++i)
            out_probs[i] *= inv_norm;
    for (i = 0, cur_out_probs = out_probs; i < size; ++i, cur_out_probs += num_classes) {
        for (j = 0; j < num_classes; ++j)
            if (leaf_prob[i] < cur_out_probs[j]) {
                leaf_prob[i] = cur_out_probs[j];
                leaf_class[i] = j;
            }
    }
    if (!all_probs)
        for (i = 0; i < size; ++i)
            leaf_prob[i] *= inv_norm;
}


int texton_feature_func(uint8_t *data, int height, int width, texton_feature_data_t *func_data, int *point_data, int point_offset, int l) {
    int i = point_data[2 * point_offset], j = point_data[2 * point_offset + 1];
    texton_feature_data_t d = func_data[l];
    int v0 = data_samp(data, i + d.uy, j + d.ux, d.ub, height, width);
    int v1;
    if (d.feat_type == 0)
        return  v0;
    v1 = data_samp(data, i + d.vy, j + d.vx, d.vb, height, width);
    if (d.feat_type == 1)
        return v0 + v1;
    else if (d.feat_type == 2)
        return v0 - v1;
    return abs(v0 - v1);
}

int hamming_distance(uint8_t v0, uint8_t v1) {
    uint8_t x = v0 ^ v1;
    int dist = 0;
    while(x) {
        ++dist;
        x &= x - 1;
    }
    return dist;
}

#define CLAMPINT(x, m, M) (((x) < (m)) ? (m) : ((x) > (M) ? (M) : (x)))
int integral_feature_func(double **data, int height, int width, integral_feature_data_t *func_data, int *point_data, int point_offset, int l) {
    int i = point_data[2 * point_offset], j = point_data[2 * point_offset + 1];
    integral_feature_data_t d = func_data[l];
    double *preds = data[0];
    double *integrals = data[1];
    if (d.pattern < 0) { // ILP case
        assert(0 <= d.ilp_dim && d.ilp_dim < d.num_ilp_dims);
        return preds[d.ilp_dim] * 2147483646;
    }
    int y0 = CLAMPINT((i + d.uy) - d.box_height_radius, 0, height - 1);
    int y1 = CLAMPINT((i + d.uy) + d.box_height_radius + 1, 0, height - 1);
    int x0 = CLAMPINT((j + d.ux) - d.box_width_radius, 0, width - 1);
    int x1 = CLAMPINT((j + d.ux) + d.box_width_radius + 1, 0, width - 1);
    double v00 = integral_samp(integrals, y0, x0, d.mask_num, height, width, d.num_masks);
    double v10 = integral_samp(integrals, y1, x0, d.mask_num, height, width, d.num_masks);
    double v01 = integral_samp(integrals, y0, x1, d.mask_num, height, width, d.num_masks);
    double v11 = integral_samp(integrals, y1, x1, d.mask_num, height, width, d.num_masks);
    // Computes the area and scales it between [0, 2147483646] (for integer discretization)
    int denom = (y1 - y0) * (x1 - x0);
    double out;
    /* Regions count in row major order of the first pixel encountered  */
    /* Integrals computed BR - BL - TR + TL */
    /* Due to the integer division in the patterns, some may have 1 more pixel */
    /* TODO(brandyn): Look into compensating for this with individual block normalizations (more divisions) */
    switch (d.pattern) {
    case 1: /* Small box centered in a big box [0] */
    {
        int xa = x0 + (x1 - x0) / 3;
        int xb = x0 + 2 * (x1 - x0) / 3;
        int ya = y0 + (y1 - y0) / 3;
        int yb = y0 + 2 * (y1 - y0) / 3;
        double vaa = integral_samp(integrals, ya, xa, d.mask_num, height, width, d.num_masks);
        double vab = integral_samp(integrals, ya, xb, d.mask_num, height, width, d.num_masks);
        double vba = integral_samp(integrals, yb, xa, d.mask_num, height, width, d.num_masks);
        double vbb = integral_samp(integrals, yb, xb, d.mask_num, height, width, d.num_masks);
        double region0 = v11 - v01 - v10 + v00;
        double region1 = vbb - vba - vab + vaa;
        out = region0 - region1;
        break;
    }
    case 2: /* Vertical stripes [|] */
    {
        int xm = x0 + (x1 - x0) / 2;
        double v0m = integral_samp(integrals, y0, xm, d.mask_num, height, width, d.num_masks);
        double v1m = integral_samp(integrals, y1, xm, d.mask_num, height, width, d.num_masks);
        double region0 = v1m - v10 - v0m + v00;
        double region1 = v11 - v1m - v01 + v0m;
        out = region1 - region0;
        break;
    }
    case 3: /* Horizontal stripes [-] */
    {
        int ym = y0 + (y1 - y0) / 2;
        double vm0 = integral_samp(integrals, ym, x0, d.mask_num, height, width, d.num_masks);
        double vm1 = integral_samp(integrals, ym, x1, d.mask_num, height, width, d.num_masks);
        double region0 = vm1 - vm0 - v01 + v00;
        double region1 = v11 - v10 - vm1 + vm0;
        out = region0 - region1;
        break;
    }
    case 4: /* Vertical stripes [||] */
    {
        int xa = x0 + (x1 - x0) / 3;
        int xb = x0 + 2 * (x1 - x0) / 3;
        double v0a = integral_samp(integrals, y0, xa, d.mask_num, height, width, d.num_masks);
        double v0b = integral_samp(integrals, y0, xb, d.mask_num, height, width, d.num_masks);
        double v1a = integral_samp(integrals, y1, xa, d.mask_num, height, width, d.num_masks);
        double v1b = integral_samp(integrals, y1, xb, d.mask_num, height, width, d.num_masks);
        double region0 = v1a - v10 - v0a + v00;
        double region1 = v1b - v1a - v0b + v0a;
        double region2 = v11 - v1b - v01 + v0b;
        out = region0 - region1 + region2;
        break;
    }
    case 5: /* Horizontal strips [=] */
    {
        int ya = y0 + (y1 - y0) / 3;
        int yb = y0 + 2 * (y1 - y0) / 3;
        double va0 = integral_samp(integrals, ya, x0, d.mask_num, height, width, d.num_masks);
        double vb0 = integral_samp(integrals, yb, x0, d.mask_num, height, width, d.num_masks);
        double va1 = integral_samp(integrals, ya, x1, d.mask_num, height, width, d.num_masks);
        double vb1 = integral_samp(integrals, yb, x1, d.mask_num, height, width, d.num_masks);
        double region0 = va1 - va0 - v01 + v00;
        double region1 = vb1 - vb0 - va1 + va0;
        double region2 = v11 - v10 - vb1 + vb0;
        out = region0 - region1 + region2;
        break;
    }
    case 6: /* Quadrants [+] */
    {
        int xm = x0 + (x1 - x0) / 2;
        int ym = y0 + (y1 - y0) / 2;
        double v0m = integral_samp(integrals, y0, xm, d.mask_num, height, width, d.num_masks);
        double vm0 = integral_samp(integrals, ym, x0, d.mask_num, height, width, d.num_masks);
        double vmm = integral_samp(integrals, ym, xm, d.mask_num, height, width, d.num_masks);
        double vm1 = integral_samp(integrals, ym, x1, d.mask_num, height, width, d.num_masks);
        double v1m = integral_samp(integrals, y1, xm, d.mask_num, height, width, d.num_masks);
        double region0 = vmm - vm0 - v0m + v00;
        double region1 = vm1 - vmm - v0m + v01;
        double region2 = v1m - v10 - vmm + vm0;
        double region3 = v11 - v1m - vm1 + vmm;
        out = region1 - region0 - region2 - region3;
        break;
    }
    case 0: /* Whole box [ ]*/
    default:
        out = v11 - v01 - v10 + v00;
    }
    return (out / denom) * 2147483646;
}

void tree_predict(int32_t *out_leaves, int tree_root, int32_t *links, int num_classes,
                  void *image_data, int height, int width, void *func_data, int32_t *threshs,
                  int (*feature_func)(void *, int, int, void *, void *, int, int)) {
    int i, j, l;
    int ij[2];
    for (i = 0; i < height; ++i)
        for (j = 0; j < width; ++j) {
            l = tree_root;
            while (l >= 0) {
                ij[0] = i;
                ij[1] = j;
                l = links[2 * l + (threshs[l] <= feature_func(image_data, height, width, func_data, ij, 0, l))];
            }
            out_leaves[width * i + j] = -(l + 1);
        }
}

void tree_predict_ijs(int32_t *ijs, int num_ijs, int32_t *out_leaves, int tree_root, int32_t *links, int num_classes,
                      void *image_data, int height, int width, void *func_data, int32_t *threshs,
                      int (*feature_func)(void *, int, int, void *, void *, int, int)) {
    int k, l;
    for (k = 0; k < num_ijs; ++k) {
        l = tree_root;
        while (l >= 0) {
            l = links[2 * l + (threshs[l] <= feature_func(image_data, height, width, func_data, ijs, k, l))];
        }
        out_leaves[k] = -(l + 1);
    }
}

void feature_func_many(void *image_data, int height, int width, void *point_data, int num_points, void *func_data, int32_t *threshs,
                       int num_threshs, uint8_t *out_bools, int (*feature_func)(void *, int, int, void *, void *, int, int)) {
    int k, l;
    int val;
    int out_offset;
    for (k = 0; k < num_points; ++k) {
        val = feature_func(image_data, height, width, func_data, point_data, k, 0);
        for (l = 0, out_offset = k; l < num_threshs; ++l, out_offset += num_points)
            out_bools[out_offset] = val >= threshs[l];
    }
}

double calc_l2_dist(double *pt1, double *pt2) { 
    return ((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])) + ((pt1[1] - pt2[1]) * (pt1[1] - pt2[1])) + ((pt1[2] - pt2[2]) * (pt1[2] - pt2[2]));
}
