import os
import imseg

try:
    feature_type = os.environ['FEATURE_TYPE']
except KeyError:
    feature_type = ''


def make_feature_factory():
    if feature_type == 'texton':
        return imseg.TextonFeatureFactory(int(os.environ['RADIUS']),
                                            num_thresh=int(os.environ['NUM_THRESH']))
    elif feature_type == 'integral':
        return imseg.IntegralFeatureFactory(num_masks=int(os.environ['NUM_INTEGRALS']),
                                              max_box_dist_radius_y=int(os.environ['MAX_BOX_DIST_RADIUS_Y']),
                                              max_box_dist_radius_x=int(os.environ['MAX_BOX_DIST_RADIUS_X']),
                                              min_box_radius_y=int(os.environ['MIN_BOX_RADIUS_Y']),
                                              min_box_radius_x=int(os.environ['MIN_BOX_RADIUS_X']),
                                              max_box_radius_y=int(os.environ['MAX_BOX_RADIUS_Y']),
                                              max_box_radius_x=int(os.environ['MAX_BOX_RADIUS_X']),
                                              num_thresh=int(os.environ['NUM_THRESH']),
                                              num_ilp_dims=int(os.environ['NUM_ILP_DIMS']),
                                              ilp_prob=float(os.environ['ILP_PROB']))
    else:
        raise ValueError('FEATURE_TYPE environmental variable not set properly!')


def make_predict(trees_ser):
    if feature_type == 'texton':
        return imseg.TextonPredict(trees_ser)
    elif feature_type == 'integral':
        return imseg.IntegralPredict(trees_ser)
    else:
        raise ValueError('FEATURE_TYPE environmental variable not set properly!')
