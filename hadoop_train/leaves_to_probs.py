import hadoopy
import imseg
import numpy as np
import os

#import picarus._classifiers as classifiers
import picarus._file_parse as file_parse
import picarus._features as features
CLASSIFIERS = {}
CLASSIFIER_FEATURE = None
ALL_CLASSIFIERS = None


def convert_leaves_all_probs_pred_old(image, leaves, all_probs, num_leaves, classifiers_fn=None):
    global CLASSIFIERS, CLASSIFIER_FEATURE
    if classifiers_fn is None:
        classifiers_fn = os.environ['CLASSIFIERS_FN']
    get_classifier_confidence = lambda x: x[0][0] * x[0][1]
    if CLASSIFIERS is None:
        all_classifiers = sorted(file_parse.load(classifiers_fn))
        name_classifiers = []
        for x in range(len(all_classifiers)):
            if x < len(all_classifiers):  # TODO(brandyn): Fix memory issue so that we can use the last classifier too
                name_classifiers.append((all_classifiers[x][0],
                                         classifiers.loads(all_classifiers[x][1])))
            else:
                name_classifiers.append((all_classifiers[x][0],
                                         name_classifiers[-1][1]))
            all_classifiers[x] = None  # NOTE(Brandyn): This is done to save memory
        print('ILP Classifiers %r' % ([x for x, _ in name_classifiers],))
        CLASSIFIERS = [x for _, x in name_classifiers]
    if CLASSIFIER_FEATURE is None:
        CLASSIFIER_FEATURE = features.select_feature('bovw_hog')
    feature = CLASSIFIER_FEATURE(np.ascontiguousarray(image[:, :, :3]))
    preds = np.ascontiguousarray([get_classifier_confidence(classifier.predict(feature))
                                  for classifier in CLASSIFIERS], dtype=np.float64)
    out0 = imseg.convert_labels_to_integrals(leaves, num_leaves)
    out1 = imseg.convert_all_probs_to_integrals(all_probs)
    return preds, np.ascontiguousarray(np.dstack([out0, out1]))


def convert_leaves_all_probs_pred(image, leaves, all_probs, num_leaves, classifiers_fn=None):
    global CLASSIFIER_FEATURE, ALL_CLASSIFIERS, CLASSIFIERS
    preds = []
    if classifiers_fn:
        get_classifier_confidence = lambda x: x[0][0] * x[0][1]
        if ALL_CLASSIFIERS is None:
            ALL_CLASSIFIERS = sorted(file_parse.load(classifiers_fn))
        if CLASSIFIER_FEATURE is None:
            CLASSIFIER_FEATURE = features.select_feature('bovw_hog')
        feature = CLASSIFIER_FEATURE(np.ascontiguousarray(image[:, :, :3]))
        for x in range(len(ALL_CLASSIFIERS)):
            try:
                classifier = CLASSIFIERS[x]
            except KeyError:
                classifier = classifiers.loads(ALL_CLASSIFIERS[x][1])
                if x < 14:
                    CLASSIFIERS[x] = classifier
            preds.append(get_classifier_confidence(classifier.predict(feature)))
    preds = np.ascontiguousarray(preds, dtype=np.float64)
    out0 = imseg.convert_labels_to_integrals(leaves, num_leaves)
    out1 = imseg.convert_all_probs_to_integrals(all_probs)
    return preds, np.ascontiguousarray(np.dstack([out0, out1]))


def predict_classifiers(image, start_ind, stop_ind, classifiers_fn=None):
    global CLASSIFIER_FEATURE, ALL_CLASSIFIERS, CLASSIFIERS
    if classifiers_fn is None:
        classifiers_fn = os.environ['CLASSIFIERS_FN']
    get_classifier_confidence = lambda x: x[0][0] * x[0][1]
    if ALL_CLASSIFIERS is None:
        ALL_CLASSIFIERS = sorted(file_parse.load(classifiers_fn))
    if CLASSIFIER_FEATURE is None:
        CLASSIFIER_FEATURE = features.select_feature('bovw_hog')
    feature = CLASSIFIER_FEATURE(np.ascontiguousarray(image[:, :, :3]))
    preds = {}
    for x in range(start_ind, stop_ind):
        try:
            classifier = CLASSIFIERS[x]
        except KeyError:
            classifier = classifiers.loads(ALL_CLASSIFIERS[x][1])
            CLASSIFIERS[x] = classifier
        preds[x] = get_classifier_confidence(classifier.predict(feature))
    return preds


def save_classifiers(hdfs_classifier_path, classes, name):
    if classes:
        classes = set(classes.split())
    file_parse.dump([x for x in hadoopy.readtb(hdfs_classifier_path)
                     if classes is None or x[0] in classes], name)
