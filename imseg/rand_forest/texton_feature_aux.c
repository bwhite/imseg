#include <stdint.h>
#include <stdlib.h>

// NOTE(brandyn): This is cast to an int so that further operations don't overflow
inline int data_samp(uint8_t *data, int i, int j, int b, int height, int width) {
    if (0 <= i && i < height && 0 <= j && j < width && 0 <= b && b < 3)
        return data[(i * width + j) * 3 + b];
    return 0;
}

inline void texton_func_many_t0(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int x0, int y0, int b0, int32_t *ts, int num_threshs, uint8_t *out_bools) {
    int k, l, m, val, out_offset;
    for (k = 0, m = 0; k < num_points; ++k, m += 2) {
        val = data_samp(data, ijs[m] + y0, ijs[m + 1] + x0, b0, height, width);
        for (l = 0, out_offset = k; l < num_threshs; ++l, out_offset += num_points)
            out_bools[out_offset] = val >= ts[l];
    }
}

inline void texton_func_many_t1(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int x0, int y0, int b0, int x1, int y1, int b1, int32_t *ts, int num_threshs, uint8_t *out_bools) {
    int k, l, m, val, out_offset;
    for (k = 0, m = 0; k < num_points; ++k, m += 2) {
        val = data_samp(data, ijs[m] + y0, ijs[m + 1] + x0, b0, height, width) + data_samp(data, ijs[m] + y1, ijs[m + 1] + x1, b1, height, width);
        for (l = 0, out_offset = k; l < num_threshs; ++l, out_offset += num_points)
            out_bools[out_offset] = val >= ts[l];
    }
}

inline void texton_func_many_t2(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int x0, int y0, int b0, int x1, int y1, int b1, int32_t *ts, int num_threshs, uint8_t *out_bools) {
    int k, l, m, val, out_offset;
    for (k = 0, m = 0; k < num_points; ++k, m += 2) {
        val = data_samp(data, ijs[m] + y0, ijs[m + 1] + x0, b0, height, width) - data_samp(data, ijs[m] + y1, ijs[m + 1] + x1, b1, height, width);
        for (l = 0, out_offset = k; l < num_threshs; ++l, out_offset += num_points)
            out_bools[out_offset] = val >= ts[l];
    }
}

inline void texton_func_many_t3(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int x0, int y0, int b0, int x1, int y1, int b1, int32_t *ts, int num_threshs, uint8_t *out_bools) {
    int k, l, m, val, out_offset;
    for (k = 0, m = 0; k < num_points; ++k, m += 2) {
        val = abs(data_samp(data, ijs[m] + y0, ijs[m + 1] + x0, b0, height, width) - data_samp(data, ijs[m] + y1, ijs[m + 1] + x1, b1, height, width));
        for (l = 0, out_offset = k; l < num_threshs; ++l, out_offset += num_points)
            out_bools[out_offset] = val >= ts[l];
    }
}

inline void texton_func_many(uint8_t *data, int height, int width, int32_t *ijs, int num_points, int feat_type, int x0, int y0, int b0, int x1, int y1, int b1, int32_t *ts, int num_threshs, uint8_t *out_bools) {
    /* Compute the texton feature between many points and many thresholds.

      Args:
          data: Row major 3 channel color
          width: Image width
          ijs: (num_points, 2)
          num_points:
          feat_type:
          x0: 
          y0:
          b0:
          x1: 
          y1:
          b1:
          ts: Thresholds (num_threshs)
          out_bools: (threshs, num_points)
    */
    if (feat_type == 0)
        texton_func_many_t0(data, height, width, ijs, num_points, x0, y0, b0, ts, num_threshs, out_bools);
    else if (feat_type == 1)
        texton_func_many_t1(data, height, width, ijs, num_points, x0, y0, b0, x1, y1, b1, ts, num_threshs, out_bools);
    else if (feat_type == 2)
        texton_func_many_t2(data, height, width, ijs, num_points, x0, y0, b0, x1, y1, b1, ts, num_threshs, out_bools);
    else
        texton_func_many_t3(data, height, width, ijs, num_points, x0, y0, b0, x1, y1, b1, ts, num_threshs, out_bools);
}

int texton_func(uint8_t *data, int height, int width, int i, int j, int feat_type, int uy, int ux, int ub, int vy, int vx, int vb, int t) {
    int v0 = data_samp(data, i + uy, j + ux, ub, height, width);
    int v1;
    if (feat_type == 0)
        return  v0 >= t;
    v1 = data_samp(data, i + vy, j + vx, vb, height, width);
    if (feat_type == 1)
        return v0 + v1 >= t;
    else if (feat_type == 2)
        return v0 - v1 >= t;
    return abs(v0 - v1) >= t;  
}

void texton_predict(uint8_t *data, int height, int width, double *out_prob, uint8_t *out_ind, int32_t *trees, int32_t *links, double *leaves,
                    int32_t *feat_type, int32_t *u, int32_t *v, int32_t *t, int num_trees, int num_classes) {
    int i, j, k, l, m;
    double *prob_sum = malloc(sizeof *prob_sum * num_classes);
    double max_prob;
    int max_prob_ind;
    double d_x_inv;
    for (i = 0; i < height; ++i)
        for (j = 0; j < width; ++j) {
            memset(prob_sum, 0, sizeof *prob_sum * num_classes);
            for (k = 0; k < num_trees; ++k) {
                l = trees[k];
                while (l >= 0) {
                    l = links[2 * l + texton_func(data, height, width, i, j, feat_type[l],
                                                  u[3 * l], u[3 * l + 1], u[3 * l + 2],
                                                  v[3 * l], v[3 * l + 1], v[3 * l + 2],
                                                  t[l])];
                }
                l = -(l + 1);
                for (m = 0; m < num_classes; ++m)
                    prob_sum[m] += leaves[num_classes * l + m];
            }
            max_prob = 0.;
            max_prob_ind = 0;
            for (m = 0; m < num_classes; ++m) {
                // NOTE: This is <= to be byte compatible with classipy's predict
                if (max_prob <= prob_sum[m]) {
                    max_prob = prob_sum[m];
                    max_prob_ind = m;
                }
            }
            out_ind[width * i + j] = max_prob_ind;
            out_prob[width * i + j] = max_prob;
        }
    free(prob_sum);
}
