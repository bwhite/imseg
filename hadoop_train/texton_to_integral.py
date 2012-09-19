import hadoopy
import imseg
import numpy as np
import tree_features
import cPickle as pickle
import os
from leaves_to_probs import convert_leaves_all_probs_pred


class Mapper(object):

    def __init__(self):
        self.tp = tree_features.make_predict(pickle.load(open(os.environ['TREES_SER_FN'])))
        self.integral_type = os.environ['INTEGRAL_TYPE']

    def map(self, k, v):
        """
        Args:
            k: Filename
            v: (image_data, label_points)

        Yields:
            (k, (integral, label_points))
        """
        image_data, label_points = v

        argmax, max_probs, leaves, all_probs = self.tp.predict(image_data,
                                                               all_probs=True,
                                                               leaves=True)
        print('Leaves[%s] AllProbs[%s]' % (str(leaves.shape), str(all_probs.shape)))
        if self.integral_type == 'argmax':
            out = np.ascontiguousarray(imseg.convert_labels_to_integrals(argmax, self.tp.num_classes))
        elif self.integral_type == 'argmax_prob':
            out = np.ascontiguousarray(imseg.convert_labels_probs_to_integrals(argmax, max_probs, self.tp.num_classes))
        elif self.integral_type == 'all_prob':
            out = np.ascontiguousarray(imseg.convert_all_probs_to_integrals(all_probs))
        elif self.integral_type == 'spatial':
            out = convert_leaves_all_probs_pred(image_data, leaves, all_probs, self.tp.num_leaves)
        yield k, (out, label_points)

if __name__ == '__main__':
    hadoopy.run(Mapper)
