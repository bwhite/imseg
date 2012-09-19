#!/usr/bin/env python
import hadoopy
import tree_features
from _imseg_rand_forest import train_map_hists, train_reduce_info, make_features
import os
import numpy as np
import cPickle as pickle
#import zlib
import snappy
import time
import contextlib
import scipy as sp
import scipy.sparse


def get_environ():
    return [int(os.environ[x]) for x in ['NUM_FEAT', 'NUM_CLASSES', 'SEED', 'LEVEL']]


class Timer(object):

    def __init__(self):
        self.total_times = {}
        self.total_counts = {}
        self.timers = {}
        self.start_time = time.time()

    def start(self, timer):
        self.timers[timer] = time.time()

    def stop(self, timer):
        diff = time.time() - self.timers[timer]
        try:
            self.total_times[timer] += diff
            self.total_counts[timer] += 1
        except KeyError:
            self.total_times[timer] = diff
            self.total_counts[timer] = 1

    @contextlib.contextmanager
    def __call__(self, timer):
        self.start(timer)
        yield
        self.stop(timer)

    def __del__(self):
        print('Timers')
        total_time = time.time() - self.start_time
        total_covered_time = 0.
        for timer_name in self.timers:
            total_covered_time += self.total_times[timer_name]
            print('%s: %f / %d = %f' % (timer_name,
                                        self.total_times[timer_name], self.total_counts[timer_name],
                                        self.total_times[timer_name] / self.total_counts[timer_name]))
        print('Total Covered: %f' % total_covered_time)
        print('Total: %f' % total_time)


class Base(object):

    def __init__(self):
        super(Base, self).__init__()
        (self.num_feat, self.num_classes, self.seed, self.level) = get_environ()
        self.feature_factory = tree_features.make_feature_factory()
        self.feats = make_features(self.feature_factory, self.num_feat,
                                   seed=self.seed)
        self.timer = Timer()

    def ser(self, a):
        return snappy.compress(pickle.dumps(a, -1))  # zlib.compress(pickle.dumps(a, -1), 9)

    def deser(self, a):
        return pickle.loads(snappy.decompress(a))  # pickle.loads(zlib.decompress(a))


class Mapper(Base):

    def __init__(self):
        super(Mapper, self).__init__()
        self.qlss, self.qrss, self.num_images = {}, {}, {}
        try:
            tree_ser = pickle.load(open(os.environ['TREE_SER_FN']))
        except KeyError:
            self.dp = None
        else:
            self.dp = tree_features.make_predict([tree_ser])
        self.min_sparsity = os.environ.get('MIN_SPARSITY', .05)
        self.max_root_buffer = os.environ.get('MAX_ROOT_BUFFER', 32)
        self.root_count = {}

    def map(self, image_name, image_label_points):
        """
        Args:
            image_name: A string (if not then we skip the input)
            image_label_points: (image, [(label, points), ...]) where points is Nx2 (y, x)
        """
        if not isinstance(image_name, str):
            hadoopy.counter('SKIPPED_INPUTS', 'KeyNotString')
            return
        with self.timer('Build root_labels_image_points'):
            image, label_points = image_label_points
            root_labels_image_points = {}
            if self.dp:
                root_label_points = self.dp.group_lowest_nodes(image, label_points)
                for root, label_points in root_label_points:
                    labels = np.array([x[0] for x in label_points], dtype=np.int32)
                    image_points = [(image, x[1]) for x in label_points]
                    root_labels_image_points[root] = labels, image_points
            else:
                labels = np.array([x[0] for x in label_points], dtype=np.int32)
                image_points = [(image, x[1]) for x in label_points]
            root_labels_image_points[0] = labels, image_points
        with self.timer('Run train_map_hists and sum qlss/qrss'):
            for root, (labels, image_points) in root_labels_image_points.items():
                if self.level != int(np.floor(np.log2(root + 1))):  # We are done processing this root
                    continue
                qls, qrs = train_map_hists(labels, image_points, self.feats, self.num_classes)
                qls, qrs = self.convert_matrix(qls), self.convert_matrix(qrs)
                try:
                    self.root_count[root] += 1
                except KeyError:
                    self.root_count[root] = 1
                try:
                    try:
                        self.qlss[root] += qls
                    except NotImplementedError:
                        self.qlss[root] = self.qlss[root] + qls
                    try:
                        self.qrss[root] += qrs
                    except NotImplementedError:
                        self.qrss[root] = self.qrss[root] + qrs
                    self.num_images[root] += 1
                except KeyError:
                    if self.max_root_buffer <= len(self.qlss):
                        for x in self.flush_node(root):
                            yield x
                    self.qlss[root] = qls
                    self.qrss[root] = qrs
                    self.num_images[root] = 1

    def convert_matrix(self, matrix):
        sparsity = len(matrix.nonzero()[0]) / float(matrix.size)
        print('Sparsity[%f]' % sparsity)
        if sparsity < self.min_sparsity:
            hadoopy.counter('SPARSITY', 'SPARSE')
            return sp.sparse.csr_matrix(matrix)
        hadoopy.counter('SPARSITY', 'DENSE')
        return matrix

    def flush_node(self, cur_node):
        least_node = None
        least_num_images = float('inf')
        print('Scanning for nodes to flush')
        for root, count in self.root_count.items():
            if root not in self.qlss:
                continue
            print('[%d] cnt[%d] num_images[%d] qlss_nz[%d/%d] qrss_nz[%d/%d]' % (root, count, self.num_images[root],
                                                                                 self.qlss[root].nonzero()[0].size, self.qlss[root].size,
                                                                                 self.qrss[root].nonzero()[0].size, self.qrss[root].size))
            if count < least_num_images and cur_node != root:
                least_num_images = count
                least_node = root
        print('Flushing [%s] as it only has [%s] images' % (least_node, least_num_images))
        yield least_node, (self.ser(self.qlss[least_node]),
                           self.ser(self.qrss[least_node]),
                           self.num_images[least_node])
        del self.qlss[least_node]
        del self.qrss[least_node]
        del self.num_images[least_node]

    def close(self):
        """
        Yields:
            Tuple of (root, ql_qr_numimage) where
            root: Root number (int)
            ql_qr_numimage: (ql, qr, num_image) which are the left/right label
            histograms for a single feature and the number of image images for
            this root.
        """
        with self.timer('Flushing remaining output'):
            for root in self.qlss:
                print('[%d] cnt[%d] num_images[%d] qlss_nz[%d/%d] qrss_nz[%d/%d]' % (root, self.root_count[root], self.num_images[root],
                                                                                     self.qlss[root].nonzero()[0].size, self.qlss[root].size,
                                                                                     self.qrss[root].nonzero()[0].size, self.qrss[root].size))
                yield root, (self.ser(self.qlss[root]),
                             self.ser(self.qrss[root]),
                             self.num_images[root])


class Combiner(Base):

    def __init__(self):
        super(Combiner, self).__init__()

    def _combine(self, root, qls_qrs_numimages):
        """
        Args:
            root: Root number (int)
            qls_qrs_numimages: (see mapper).

        Yields:
            Tuple of (root, qls_qrs_numimages) where
            root: Integer root number
            qls_qrs_numimages: Iterator of ql_qr_numimage (see mapper)
        """
        qls_sum, qrs_sum, num_image_sum = qls_qrs_numimages.next()
        qls_sum, qrs_sum = self.deser(qls_sum), self.deser(qrs_sum)
        for qls, qrs, num_image in qls_qrs_numimages:
            qls = self.deser(qls)
            qrs = self.deser(qrs)
            try:
                qls_sum += qls
            except NotImplementedError:
                qls_sum = qls_sum + qls
            try:
                qrs_sum += qrs
            except NotImplementedError:
                qrs_sum = qrs_sum + qrs
            num_image_sum += num_image
        yield root, (qls_sum, qrs_sum, num_image_sum)

    def reduce(self, root, qls_qrs_numimages):
        """
        Args:
            root_feat_num: String of 'root\tfeat_num'
            ql_qr_numimages: Iterator of ql_qr_numimage (see mapper)

        Yields:
            Tuple of (root, value) where
            root: Integer node number
            value: (qls, qrs, num_image) (see mapper)
        """
        root, (qls, qrs, num_image) = self._combine(root, qls_qrs_numimages).next()
        yield root, (self.ser(qls), self.ser(qrs), num_image)


class Reducer(Combiner):

    def __init__(self):
        super(Reducer, self).__init__()

    def reduce(self, root, qls_qrs_numimages):
        """
        Args:
            root_feat_num: String of 'root\tfeat_num'
            ql_qr_numimages: Iterator of ql_qr_numimage (see mapper)

        Yields:
            Tuple of (root, value) where
            root: Integer node number
            value: (info_gain, qls, qrs, feat_ser, num_image)
        """
        _, (qls, qrs, num_image) = self._combine(root, qls_qrs_numimages).next()
        try:
            qls = qls.toarray()
        except AttributeError:
            qls = np.asarray(qls)  # Must already be an array and not sparse
        try:
            qrs = qrs.toarray()
        except AttributeError:
            qrs = np.asarray(qrs)  # Must already be an array and not sparse
        info_gain, feat_num = train_reduce_info(qls, qrs)
        feat_ser = self.feature_factory.select_feature(self.feats, feat_num).dumps()
        yield root, (info_gain, qls[feat_num], qrs[feat_num],
                     feat_ser, num_image)

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer)
