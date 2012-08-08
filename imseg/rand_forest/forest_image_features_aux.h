#ifndef FOREST_IMAGE_PREDICT_AUX
#define FOREST_IMAGE_PREDICT_AUX
#include <stdint.h>

typedef struct texton_feature_data {
    int32_t feat_type;
    int32_t uy;
    int32_t ux;
    int32_t ub;
    int32_t vy;
    int32_t vx;
    int32_t vb;
} texton_feature_data_t;

typedef struct integral_feature_data {
    int32_t mask_num;
    int32_t box_height_radius;
    int32_t box_width_radius;
    int32_t uy;
    int32_t ux;
    int32_t num_masks;
    int32_t pattern;
    int32_t num_ilp_dims;
    int32_t ilp_dim;
} integral_feature_data_t;

void sum_thresh_array_fast(uint8_t *thresh_mask, int num_threshs, int num_points, int32_t *out_false, int32_t *out_true);
int texton_feature_func(uint8_t *data, int height, int width, texton_feature_data_t *func_data, int *point_data, int point_offset, int l);
int integral_feature_func(double **data, int height, int width, integral_feature_data_t *func_data, int *point_data, int point_offset, int l);

void tree_predict(int32_t *out_leaves, int tree_root, int32_t *links, int num_classes,
                  void *image_data, int height, int width, void *func_data, int32_t *threshs,
                  int (*feature_func)(void *, int, int, void *, void *, int, int));
void tree_predict_ijs(int32_t *ijs, int num_ijs, int32_t *out_leaves, int tree_root, int32_t *links, int num_classes,
                      void *image_data, int height, int width, void *func_data, int32_t *threshs,
                      int (*feature_func)(void *, int, int, void *, void *, int, int));
void feature_func_many(void *image_data, int height, int width, void *point_data, int num_points, void *func_data, int32_t *threshs,
                       int num_threshs, uint8_t *out_bools, int (*feature_func)(void *, int, int, void *, void *, int, int));
void predict_multi_tree_max_argmax(int size, int num_classes, int num_trees, int all_probs, int32_t *out_leaves, double *leaves,
                                   double *out_probs, int32_t *leaf_class, double *leaf_prob);
#endif
