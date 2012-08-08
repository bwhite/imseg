#ifndef TEXTON_FEATURE_AUX_H
#define TEXTON_FEATURE_AUX_H
#include <stdint.h>
void texton_func_many(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int feat_type, int x0, int y0, int b0, int x1, int y1, int b1, int32_t *ts, int num_threshs, uint8_t *out_bools);
void texton_predict(uint8_t *data, int height, int width, double *out_prob, uint8_t *out_ind, int32_t *trees, int32_t *links, double *leaves,
                    int32_t *feat_type, int32_t *u, int32_t *v, int32_t *t, int num_trees, int num_classes);
#endif
