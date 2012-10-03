# (C) Copyright 2012 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
## cython: profile=True
"""Random Forest Classifier
See: http://research.microsoft.com/pubs/145347/BodyPartRecognition.pdf
and: http://cvlab.epfl.ch/~lepetit/papers/lepetit_cvpr05.pdf
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import numpy as np
cimport numpy as np
import operator
import cPickle as pickle
import imseg
cimport cython
import cython

cdef extern from "forest_image_features_aux.h":
    void sum_thresh_array_fast(np.uint8_t *thresh_mask, int num_threshs, int num_points, np.int32_t *out_false, np.int32_t *out_true)
    void tree_predict(np.int32_t *out_leaves, int tree_root, np.int32_t *links, int num_classes,
                      void *image_data, int height, int width, void *func_data, np.int32_t *threshs,
                      int (*feature_func)(void *, int, int, void *, void *, int, int))
    void tree_predict_ijs(np.int32_t *ijs, int num_ijs, np.int32_t *out_leaves, int tree_root, np.int32_t *links, int num_classes,
                      void *image_data, int height, int width, void *func_data, np.int32_t *threshs,
                          int (*feature_func)(void *, int, int, void *, void *, int, int))
    int texton_feature_func(void *data, int height, int width, void *func_data, void *point_data, int point_offset, int l)
    int integral_feature_func(void *data, int height, int width, void *func_data, void *point_data, int point_offset, int l)
    void feature_func_many(void *image_data, int height, int width, void *point_data, int num_points, void *func_data, np.int32_t *threshs,
                           int num_threshs, np.uint8_t *out_bools, int (*feature_func)(void *, int, int, void *, void *, int, int))
    void predict_multi_tree_max_argmax(int size, int num_classes, int num_trees, int all_probs, np.int32_t *out_leaves,
                                       np.float64_t *leaves, np.float64_t *out_probs, np.int32_t *leaf_class, np.float64_t *leaf_prob)


cdef class BaseForestFeatureFactory(object):
    cdef float max_dist
    cdef double min_thresh
    cdef double max_thresh
    cdef int num_thresh
    cdef object feature_class

    def __init__(self, min_thresh, max_thresh, num_thresh=50, max_dist=0., feature_class=None):
        self.max_dist = max_dist
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.num_thresh = num_thresh
        self.feature_class = feature_class

    cdef _rand_2d_vec(self):
        cdef float angle = np.random.random() * 2 * np.pi
        return self.max_dist * np.random.random() * np.array([np.cos(angle), np.sin(angle)])

    cdef _rand_u_v_5dof(self):
        return self._rand_2d_vec(), self._rand_2d_vec()

    cdef _rand_u_v_3dof(self):
        cdef float angle0 = np.random.random() * 2 * np.pi
        cdef float angle1 = angle0 + np.pi
        u = self.max_dist * np.random.random() * np.array([np.cos(angle0), np.sin(angle0)])
        v = self.max_dist * np.random.random() * np.array([np.cos(angle1), np.sin(angle1)])
        return u, v

    def _rand_threshs_int32(self, min_thresh=None, max_thresh=None):
        if min_thresh is None:
            min_thresh = self.min_thresh
        if max_thresh is None:
            max_thresh = self.max_thresh
        return np.array(np.random.randint(min_thresh, max_thresh + 1,
                                          self.num_thresh), dtype=np.int32)

    def _rand_threshs(self):
        return np.random.uniform(self.min_thresh, self.max_thresh, self.num_thresh)

    def gen_feature(self):
        raise NotImplementedError
 
    def loads(self, feat_ser):
        return self.feature_class(feat_ser=feat_ser)

    def select_feature(self, feats, feat_ind):
        """Select a feature by index

        This is used because each feature may have many internal configurations

        Args:
            feats: List of features
            feat_ind: Integer feature index

        Return:
            Feature
        """
        return feats[feat_ind / self.num_thresh][feat_ind % self.num_thresh]

    def leaf_probability(self, labels, values, num_classes):
        p = imseg.rand_forest.histogram_weight(labels, np.array([[len(x[1]) for x in values]], dtype=np.int32),
                                                  num_classes).reshape(num_classes)
        return p / float(np.sum(p))


cdef class TextonFeatureFactory(BaseForestFeatureFactory):
    cdef int region_radius

    def __init__(self, region_radius, max_thresh=255, num_thresh=50):
        super(TextonFeatureFactory, self).__init__(-max_thresh, max_thresh, num_thresh, feature_class=TextonFeature)
        self.region_radius = region_radius

    def gen_feature(self):
        feat_type = np.random.randint(0, 4)
        threshs = self._rand_threshs_int32()
        if feat_type == 0:  # Single pixel value: x,y,b
            x, y = np.random.randint(-self.region_radius, self.region_radius + 1, 2)
            b = np.random.randint(0, 7)  # RGB + LAB + Gradient = [0, 7)
            return TextonFeature(feat_type=0, x0=x, y0=y, b0=b, x1=0, y1=0, b1=0, threshs=threshs)
        else:
            x0, y0, x1, y1 = np.random.randint(-self.region_radius, self.region_radius + 1, 4)  # NOTE(brandyn): These shouldn't equal
            b0 = np.random.randint(0, 7)  # RGB + LAB + Gradient = [0, 7)
            b1 = np.random.randint(0, 7)
            return TextonFeature(feat_type=feat_type,
                                 x0=x0, y0=y0, b0=b0,
                                 x1=x1, y1=y1, b1=b1,
                                 threshs=threshs)


cdef class IntegralFeatureFactory(BaseForestFeatureFactory):
    cdef int num_masks, max_box_dist_radius_y, max_box_dist_radius_x, min_box_radius_y, min_box_radius_x, max_box_radius_y, max_box_radius_x, num_patterns, num_ilp_dims
    cdef double ilp_prob
    cdef object thresh_bounds

    def __init__(self, num_masks, max_box_dist_radius_y=30, max_box_dist_radius_x=30, min_box_radius_y=5, min_box_radius_x=5, max_box_radius_y=50,
                 max_box_radius_x=50, max_thresh=2147483646, num_patterns=6, num_ilp_dims=0, ilp_prob=.1, num_thresh=100):
        super(IntegralFeatureFactory, self).__init__(0, max_thresh, num_thresh, feature_class=IntegralFeature)
        self.num_masks = num_masks
        self.max_box_dist_radius_y = max_box_dist_radius_y
        self.max_box_dist_radius_x = max_box_dist_radius_x
        self.min_box_radius_y = min_box_radius_y
        self.min_box_radius_x = min_box_radius_x
        self.max_box_radius_y = max_box_radius_y
        self.max_box_radius_x = max_box_radius_x
        self.num_patterns = num_patterns
        self.num_ilp_dims = num_ilp_dims
        if num_ilp_dims:
            self.ilp_prob = ilp_prob
        else:
            self.ilp_prob = 0.
        self.thresh_bounds = np.array(np.array([(0, 1),
                                                (-1/8., 7/8.),
                                                (-1/2., 1/2.),
                                                (-1/2., 1/2.),
                                                (-1/3., 2/3.),
                                                (-1/3., 2/3.),
                                                (-1/2., 1/2.)]) * max_thresh, dtype=np.int32)

    def gen_feature(self):
        if self.num_ilp_dims and np.random.random() < self.ilp_prob:
            pattern = -1
            ilp_dim = np.random.randint(0, self.num_ilp_dims)
            threshs = self._rand_threshs_int32(0, self.max_thresh)
            mask_num = 0
            box_height_radius = 0
            box_width_radius = 0
            uy = 0
            ux = 0
        else:
            pattern = np.random.randint(0, self.num_patterns)
            ilp_dim = 0
            threshs = self._rand_threshs_int32(*self.thresh_bounds[pattern])
            mask_num = np.random.randint(0, self.num_masks)
            box_height_radius = np.random.randint(self.min_box_radius_y, self.max_box_radius_y + 1)
            box_width_radius = np.random.randint(self.min_box_radius_x, self.max_box_radius_x + 1)
            uy = np.random.randint(-self.max_box_dist_radius_y, self.max_box_dist_radius_y + 1)
            ux = np.random.randint(-self.max_box_dist_radius_x, self.max_box_dist_radius_x + 1)
        return IntegralFeature(mask_num=mask_num, box_height_radius=box_height_radius, box_width_radius=box_width_radius,
                               uy=uy, ux=ux, threshs=threshs,
                               num_masks=self.num_masks,
                               pattern=pattern,
                               num_ilp_dims=self.num_ilp_dims,
                               ilp_dim=ilp_dim)

def _convert_image_data(image_data):
    return image_data, (image_data.shape[0], image_data.shape[1]), ()


def convert_classifier_integral(image_data):
    cdef np.ndarray preds, integrals
    preds = np.ascontiguousarray(image_data[0], dtype=np.float64)
    integrals = np.ascontiguousarray(image_data[1], dtype=np.float64)
    height_width = (integrals.shape[0], integrals.shape[1])
    #out_data = np.ascontiguousarray([<int>preds.data, <int>integrals.data], dtype=np.int)
    if cython.sizeof(cython.p_void) == 4:
        out_data = np.ascontiguousarray([<np.uint32_t>preds.data, <np.uint32_t>integrals.data], dtype=np.uint32)
    elif cython.sizeof(cython.p_void) == 8:
        out_data = np.ascontiguousarray([<np.uint64_t>preds.data, <np.uint64_t>integrals.data], dtype=np.uint64)
    else:
        raise ValueError('Unexpected void pointer size! [%d]' % cython.sizeof(cython.p_void))
    return out_data, height_width, (preds, integrals, image_data)


cdef class BaseRandomForestFeature(object):
    cdef public np.ndarray threshs
    cdef np.ndarray func_data  # [nodes] for all nodes
    cdef object extra_dump_vars
    cdef int (*feature_func)(void *, int, int, void *, void *, int, int)
    cdef public object convert_image_data

    def __init__(self, extra_dump_vars=(), convert_image_data=None, **kw):
        self.extra_dump_vars = extra_dump_vars
        if 'feat_ser' in kw:
            self._deserialize(kw['feat_ser'])
        else:
            self._init_from_dict(kw)
        self.convert_image_data = convert_image_data if convert_image_data else _convert_image_data

    def _init_from_dict(self, d):
        for var_name, var_type in self.extra_dump_vars:
            setattr(self, var_name, d[var_name])
        self.threshs = d['threshs']
        dtype = np.dtype(self.extra_dump_vars, align=True)
        vals = tuple([d[x] for x, _ in self.extra_dump_vars])
        self.func_data = np.array([vals], dtype=dtype)
        
    def _deserialize(self, feat_ser):
        self._init_from_dict(pickle.loads(feat_ser))

    def __str__(self):
        t = self.threshs[0][0] if self.threshs.size == 1 else self.threshs
        s = ', '.join(['%s:%s' % (x, getattr(self, x)) for x, _ in self.extra_dump_vars])
        return '%s <= [%s]' % (t, s)

    def dumps(self):
        return pickle.dumps(dict([(x, getattr(self, x))
                                  for x, _ in self.extra_dump_vars + [('threshs', '')]]), -1)

    def __repr__(self):
        s = ', '.join(['%s=%s' % (x, getattr(self, x)) for x, _ in self.extra_dump_vars])
        if s:
            s += ', '
        return '%s(%sthreshs=%r)' % (self.__class__.__name__, s, self.threshs)

    def __getitem__(self, index):
        kw = dict([(x, getattr(self, x)) for x, _ in self.extra_dump_vars])
        return self.__class__(threshs=np.array([[self.threshs.flat[int(index)]]]), **kw)

    def __call__(self, image_ijs):
        """
        Args:
            image_ijs: Image and ijs arrays in a tuple
        
        Returns:
            Boolean array where neg/pos_inds are of shape (num_thresh, num_points)
        """
        cdef np.ndarray image
        cdef np.ndarray ijs
        cdef np.ndarray out_bools
        image_data, ijs = image_ijs
        image, height_width, _holder = self.convert_image_data(image_data)
        num_points = ijs.shape[0]
        num_threshs = self.threshs.size
        out_bools = np.zeros(num_threshs * num_points, dtype=np.bool)
        feature_func_many(<void *>image.data, height_width[0], height_width[1], <np.int32_t *>ijs.data, num_points,
                        <void *>self.func_data.data, <np.int32_t *>self.threshs.data,
                        num_threshs, <np.uint8_t *>out_bools.data, self.feature_func)
        return out_bools.reshape((num_threshs, num_points))
    
    cdef sum_thresh_array(self, np.ndarray[np.uint8_t, ndim=2, mode='c', cast=True]  thresh_mask):
        cdef np.ndarray out_false = np.zeros(thresh_mask.shape[0], dtype=np.int32)
        cdef np.ndarray out_true = np.zeros(thresh_mask.shape[0], dtype=np.int32)
        sum_thresh_array_fast(<np.uint8_t *>thresh_mask.data, thresh_mask.shape[0], thresh_mask.shape[1],
                              <np.int32_t *>out_false.data, <np.int32_t *>out_true.data)
        return out_false, out_true
    
    cpdef label_histograms(self, labels, values, int num_classes):
        """
        Args:
            labels: np.array of ints
            values: np.array of vectors

        Returns:
            Tuple of (qls, qrs)
            qls: List of elements of vecs s.t. func is false
            qrs: List of elements of vecs s.t. func is true
        """
        #cdef thresh_sums_left
        #cdef thresh_sums_right
        cdef int val_ind
        thresh_sums_left = np.zeros((len(self.threshs), len(values)), dtype=np.int32)
        thresh_sums_right = np.zeros((len(self.threshs), len(values)), dtype=np.int32)
        for val_ind, value in enumerate(values):
            thresh_sum_left, thresh_sum_right = self.sum_thresh_array(self(value))
            thresh_sums_left[:, val_ind] = thresh_sum_left
            thresh_sums_right[:, val_ind] = thresh_sum_right
        qls = imseg.rand_forest.histogram_weight(labels, thresh_sums_left, num_classes)
        qrs = imseg.rand_forest.histogram_weight(labels, thresh_sums_right, num_classes)
        return qls, qrs

    cpdef label_values_partition(self, labels, values):
        """Only uses the first row of values, producing 1 partition

        Args:
            labels: Iterator of ints
            values: Iterator of tuples with the last value as ijs (the rest are arbitrary).
                The leading values in the tuple could be an image, mask, or some other data
                that is required to compute the feature.

        Returns:
            Tuple of (ql_lab, ql_val, qr_lab, qr_val)
            ql: Elements of vecs s.t. func is false
            qr: Elements of vecs s.t. func is true
        """
        cdef np.ndarray ijs
        cdef np.ndarray mask
        #cdef np.ndarray thresh_masks
        ql_lab, qr_lab = [], []
        ql_val, qr_val = [], []
        for label, value in zip(labels, values):
            thresh_masks = self(value)
            ijs = value[-1]
            mask = thresh_masks[0]
            # Split ij's between the branches                    
            if not np.all(mask): # Left branch
                ql_lab.append(label)
                ql_val.append(tuple(list(value[:-1]) + [ijs[~mask]]))
            if np.any(mask):  # Right branch
                qr_lab.append(label)
                qr_val.append(tuple(list(value[:-1]) + [ijs[mask]]))
        return (np.ascontiguousarray(ql_lab), ql_val,
                np.ascontiguousarray(qr_lab), qr_val)


cdef class TextonFeature(BaseRandomForestFeature):
    cdef public int feat_type, y0, x0, b0, y1, x1, b1

    def __init__(self, **kw):
        super(TextonFeature, self).__init__([('feat_type', 'i4'),
                                             ('y0', 'i4'), ('x0', 'i4'), ('b0', 'i4'),
                                             ('y1', 'i4'), ('x1', 'i4'), ('b1', 'i4')], **kw)
        self.feature_func = texton_feature_func

    def __str__(self):
        t = self.threshs[0][0] if self.threshs.size == 1 else self.threshs
        if self.feat_type == 0:
            return '%s <= [x:%d, y:%d, b:%d]' % (t, self.x0, self.y0, self.b0)
        elif self.feat_type == 1:
            return '%s <= [x:%d, y:%d, b:%d] + [x:%d, y:%d, b:%d]' % (t, self.x0, self.y0, self.b0, self.x1, self.y1, self.b1)
        elif self.feat_type == 2:
            return '%s <= [x:%d, y:%d, b:%d] - [x:%d, y:%d, b:%d]' % (t, self.x0, self.y0, self.b0, self.x1, self.y1, self.b1)
        else:
            return '%s <= |[x:%d, y:%d, b:%d] - [x:%d, y:%d, b:%d]|' % (t, self.x0, self.y0, self.b0, self.x1, self.y1, self.b1)


cdef class IntegralFeature(BaseRandomForestFeature):
    cdef public np.int32_t mask_num, box_height_radius, box_width_radius, uy, ux, num_masks, pattern, num_ilp_dims, ilp_dim

    def __init__(self, **kw):
        super(IntegralFeature, self).__init__([('mask_num', 'i4'), ('box_height_radius', 'i4'), ('box_width_radius', 'i4'),
                                               ('uy', 'i4'), ('ux', 'i4'), ('num_masks', 'i4'), ('pattern', 'i4'),
                                               ('num_ilp_dims', 'i4'), ('ilp_dim', 'i4')], convert_image_data=convert_classifier_integral, **kw)
        self.feature_func = integral_feature_func


cdef class BaseForestPredict(object):
    cdef public object trees_ser
    # Below are used for updating the trees
    cdef int node_counter
    cdef int leaf_counter
    cdef object temp_leaves
    cdef object temp_links
    cdef object temp_node_nums
    cdef public object temp_t
    # Below are used for storing the trees
    cdef np.ndarray trees # [trees]
    cdef np.ndarray leaves  # [leaves, num_classes]
    cdef np.ndarray links  # [nodes, 2] false/true paths
    cdef np.ndarray leaf_class # [leaves]
    cdef np.ndarray leaf_prob # [leaves]
    cdef np.ndarray node_nums # [leaves]
    cdef public np.ndarray t  # [nodes] for all nodes
    cdef public np.ndarray func_data  # [nodes] for all nodes
    cdef public int num_trees
    cdef public int num_nodes
    cdef public int num_leaves
    cdef public int num_classes
    cdef object dump_vars
    cdef public object extra_dump_vars
    cdef int (*feature_func)(void *, int, int, void *, void *, int, int)
    cdef public object convert_image_data

    def __init__(self, trees_ser, extra_dump_vars, convert_image_data=_convert_image_data):
        self.trees_ser = trees_ser
        self.dump_vars = ['leaves', 'links', 'num_classes'] + extra_dump_vars
        self.extra_dump_vars = extra_dump_vars
        self.convert_image_data = convert_image_data

    cpdef dump(self):
        return dict([(x, getattr(self, x)) for x in self.dump_vars])

    cdef make_feature_func(self, feat_str):
        raise NotImplementedError

    cpdef update_trees(self, trees_ser):
        self.node_counter = 0
        self.leaf_counter = 0
        self.temp_leaves = []
        self.temp_links = []
        self.temp_node_nums = []
        self.temp_t = []
        self.trees = np.array([self.tree_deserialize(x, 0)
                               for x in trees_ser], dtype=np.int32)
        self.leaves = np.array(self.temp_leaves, dtype=np.float64)
        self.links = np.array(self.temp_links, dtype=np.int32)
        self.t = np.array(self.temp_t, dtype=np.int32)
        self.node_nums = np.array(self.temp_node_nums, dtype=np.int32)
        self.num_trees = len(self.trees)
        self.num_nodes = len(self.temp_t)
        self.num_leaves = len(self.temp_leaves)
        self.num_classes = len(self.temp_leaves[0])
        self.leaf_class = np.argmax(self.leaves, axis=1)
        self.leaf_prob = np.max(self.leaves, axis=1)

    cpdef tree_deserialize(self, tree_ser, node_num):
        """Given a tree_ser, gives back a tree

        Args:
            tree_ser: Tree of the form (recursive)
                (func_ser, left_tree(false), right_tree(true), metadata)
                until the leaf nodes which are (prob array, )
            node_num: The number of the current node

        Returns:
            Same structure except func_ser is converted to func using
            make_feature_func.
        """
        if len(tree_ser) != 4:
            val = self.leaf_counter
            self.leaf_counter += 1
            self.temp_leaves.append(tree_ser[0])
            self.temp_node_nums.append(node_num)
            assert val + 1 == len(self.temp_leaves)
            return -val - 1
        val = self.node_counter
        self.node_counter += 1
        for var_name, var_val in zip(self.extra_dump_vars, self.make_feature_func(tree_ser[0])):
            getattr(self, 'temp_%s' % var_name).append(var_val)
        self.temp_links.append([])
        assert val + 1 == len(self.temp_links)
        self.temp_links[val] = [self.tree_deserialize(tree_ser[1], node_num * 2 + 1),
                                self.tree_deserialize(tree_ser[2], node_num * 2 + 2)]
        return val
    
    cpdef predict(self, image_data, leaves=False, all_probs=False):
        image_data, height_width, _holder = self.convert_image_data(image_data)
        if len(self.trees) == 1:
            return self._predict_single_tree(image_data, leaves, all_probs, height_width)
        assert leaves == False
        return self._predict_multi_tree(image_data, all_probs, height_width)

    cpdef _predict_multi_tree(self, image_data, all_probs, height_width):
        cdef np.ndarray image = image_data
        height, width = height_width
        cdef np.ndarray out_leaves = np.zeros((len(self.trees), height, width), dtype=np.int32)
        cdef np.ndarray cur_out_leaves
        cdef np.ndarray out_probs = np.zeros((height, width, self.num_classes), dtype=np.float64)
        cdef np.ndarray leaf_class = np.zeros(height_width, dtype=np.int32)
        cdef np.ndarray leaf_prob = np.zeros(height_width, dtype=np.float64)
        for tree_num, tree in enumerate(self.trees):
            cur_out_leaves = out_leaves[tree_num]
            tree_predict(<np.int32_t *>cur_out_leaves.data, tree, <np.int32_t *>self.links.data, self.num_classes,
                         <void *>image.data, height_width[0], height_width[1], <void *>self.func_data.data,
                         <np.int32_t *>self.t.data, self.feature_func)
        predict_multi_tree_max_argmax(out_probs.shape[0] * out_probs.shape[1], out_probs.shape[2], len(self.trees), int(all_probs),
                                      <np.int32_t *>out_leaves.data, <np.float64_t *>self.leaves.data,
                                      <np.float64_t *>out_probs.data, <np.int32_t *>leaf_class.data,
                                      <np.float64_t *>leaf_prob.data)
        return (leaf_class, leaf_prob, out_probs) if all_probs else (leaf_class, leaf_prob)

    cpdef _predict_single_tree(self, image_data, leaves, all_probs, height_width):
        cdef np.ndarray image = image_data
        cdef np.ndarray out_leaves = np.zeros((height_width[0], height_width[1]), dtype=np.int32)
        tree_predict(<np.int32_t *>out_leaves.data, self.trees[0], <np.int32_t *>self.links.data, self.num_classes,
                     <void *>image.data, height_width[0], height_width[1], <void *>self.func_data.data,
                     <np.int32_t *>self.t.data, self.feature_func)
        if all_probs:
            p = self.leaves[out_leaves, :]
            if not leaves:
                return self.leaf_class[out_leaves], self.leaf_prob[out_leaves], p
            else:
                return self.leaf_class[out_leaves], self.leaf_prob[out_leaves], out_leaves, p
        if leaves:
            return self.leaf_class[out_leaves], self.leaf_prob[out_leaves], out_leaves
        return self.leaf_class[out_leaves], self.leaf_prob[out_leaves]
        

    cpdef group_lowest_nodes(self, image_data, label_points, int tree_num=0):
        """
        Args:
            image_data: image
            label_points: [(label, points)] where points is Nx2 in the form of (i, j)
            tree_num: Tree number to consider

        Returns:
            List of (root, label_points)
        """
        cdef np.ndarray out_nodes
        cdef np.ndarray points
        cdef np.ndarray image
        cdef np.ndarray out_leaves
        image, height_width, _holder = self.convert_image_data(image_data)
        root_label_points = {}  # [root][label] = [ijs, ...]
        for label, points in label_points:
            out_leaves = np.zeros(points.shape[0], dtype=np.int32)
            tree_predict_ijs(<np.int32_t *>points.data, points.shape[0], <np.int32_t *>out_leaves.data, self.trees[tree_num],
                             <np.int32_t *>self.links.data, self.num_classes,
                             <void *>image.data, height_width[0], height_width[1], <void *>self.func_data.data,
                             <np.int32_t *>self.t.data, self.feature_func)
            out_nodes = self.node_nums[out_leaves]
            for out_node, point in zip(out_nodes, points):
                root_label_points.setdefault(out_node, {}).setdefault(label, []).append(point)
        return [(root, [(x, np.ascontiguousarray(np.vstack(y)))
                        for x, y in z.items()])
                for root, z in root_label_points.items()]


cdef class TextonPredict(BaseForestPredict):
    cdef public object temp_feat_type, temp_x0, temp_y0, temp_b0, temp_x1, temp_y1, temp_b1

    def __init__(self, trees_ser):
        super(TextonPredict, self).__init__(trees_ser, ['feat_type', 'y0', 'x0', 'b0', 'y1', 'x1', 'b1', 't'])
        self.update_trees(trees_ser)
        self.feature_func = texton_feature_func

    cdef make_feature_func(self, feat_str):
        data = pickle.loads(feat_str)
        return data['feat_type'], data['y0'], data['x0'], data['b0'], data['y1'], data['x1'], data['b1'], data['threshs'].flat[0]

    cpdef update_trees(self, trees_ser):
        self.temp_feat_type, self.temp_y0, self.temp_x0, self.temp_b0, self.temp_y1, self.temp_x1, self.temp_b1 = [], [], [], [], [], [], []
        BaseForestPredict.update_trees(self, trees_ser)  # NOTE(brandyn): Work around for cython recursion bug
        feat_type = np.array(self.temp_feat_type, dtype=np.int32)
        y0 = np.array(self.temp_y0, dtype=np.int32)
        x0 = np.array(self.temp_x0, dtype=np.int32)
        b0 = np.array(self.temp_b0, dtype=np.int32)
        y1 = np.array(self.temp_y1, dtype=np.int32)
        x1 = np.array(self.temp_x1, dtype=np.int32)
        b1 = np.array(self.temp_b1, dtype=np.int32)
        self.func_data = np.array(zip(feat_type, y0, x0, b0, y1, x1, b1), dtype='i4,i4,i4,i4,i4,i4,i4')


cdef class IntegralPredict(BaseForestPredict):
    cdef public object temp_mask_num, temp_box_height_radius, temp_box_width_radius, temp_uy, temp_ux, temp_num_masks, temp_pattern, temp_num_ilp_dims, temp_ilp_dim

    def __init__(self, trees_ser):
        super(IntegralPredict, self).__init__(trees_ser, ['mask_num', 'box_height_radius', 'box_width_radius', 'uy', 'ux', 'num_masks',
                                                          'pattern', 'num_ilp_dims', 'ilp_dim', 't'],
                                              convert_image_data=convert_classifier_integral)
        self.update_trees(trees_ser)
        self.feature_func = integral_feature_func

    cdef make_feature_func(self, feat_str):
        data = pickle.loads(feat_str)
        return (data['mask_num'], data['box_height_radius'], data['box_width_radius'], data['uy'], data['ux'],
                data['num_masks'], data['pattern'], data['num_ilp_dims'], data['ilp_dim'], data['threshs'].flat[0])

    cpdef update_trees(self, trees_ser):
        self.temp_mask_num, self.temp_box_height_radius, self.temp_box_width_radius, self.temp_uy, self.temp_ux, self.temp_num_masks, self.temp_pattern, self.temp_num_ilp_dims, self.temp_ilp_dim = [], [], [], [], [], [], [], [], []
        BaseForestPredict.update_trees(self, trees_ser)  # NOTE(brandyn): Work around for cython recursion bug
        mask_num = np.array(self.temp_mask_num, dtype=np.int32)
        box_height_radius = np.array(self.temp_box_height_radius, dtype=np.int32)
        box_width_radius = np.array(self.temp_box_width_radius, dtype=np.int32)
        uy = np.array(self.temp_uy, dtype=np.int32)
        ux = np.array(self.temp_ux, dtype=np.int32)
        num_masks = np.array(self.temp_num_masks, dtype=np.int32)
        pattern = np.array(self.temp_pattern, dtype=np.int32)
        num_ilp_dims = np.array(self.temp_num_ilp_dims, dtype=np.int32)
        ilp_dim = np.array(self.temp_ilp_dim, dtype=np.int32)
        self.func_data = np.array(zip(mask_num, box_height_radius, box_width_radius, uy, ux, num_masks, pattern, num_ilp_dims, ilp_dim),
                                  dtype='i4,i4,i4,i4,i4,i4,i4,i4,i4')
