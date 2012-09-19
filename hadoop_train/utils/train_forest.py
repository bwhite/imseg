import kontort
#import classipy
import numpy as np
import cPickle as pickle
import os
import cv2
import random
import matplotlib.pyplot as mp
import time
import hadoopy
import imfeat
import cv
import gzip
from leaves_to_probs import convert_leaves_all_probs_pred, save_classifiers

radius = 10


rnd = lambda : np.array(np.random.randint(0, 255, 1000), dtype=np.uint8)
texton_colors = np.ascontiguousarray(np.array([rnd(), rnd(), rnd()]).T)

colors = np.asarray([[97, 37, 193], [35, 139, 35], [185, 199, 213], [20, 255, 100],
                     [19, 64, 139], [180, 180, 180], [165, 62, 163], [200, 214, 50],
                     [255, 20, 25], [201, 201, 205], [112, 25, 25], [99, 231, 9],
                     [131, 148, 32], [21, 183, 156], [201, 119, 122], [78, 63, 141],
                     [171, 7, 97], [43, 200, 126], [34, 189, 202], [207, 47, 46], [129, 176, 213]])  # NOTE (brandyn): BGR


msrc_classes = {'building': 9, 'sheep': 11, 'flower': 19, 'bicycle': 7, 'cow': 15, 'face': 13, 'sky': 8, 'tree': 1, 'dog': 4, 'sign': 0, 'water': 10, 'book': 20, 'body': 12, 'cat': 6, 'boat': 14, 'aeroplane': 17, 'car': 18, 'chair': 16, 'grass': 3, 'bird': 2, 'road': 5}
sun09_classes = dict((y, x) for x, y in enumerate(sorted('building tree sky road ground plant grass window person door wall car mountain fence table chair cupboard bed bottle flowers water'.split())))
attribute_classes = dict((y, x) for x, y in enumerate(['blue', 'brown', 'gray', 'wooden', 'furry', 'wet', 'smooth', 'yellow', 'metallic', 'rough', 'pink', 'black', 'vegetation', 'violet', 'orange', 'green', 'white', 'shiny', 'red']))

def save_key():
    class_names = [x[1] for x in sorted([x[::-1] for x in classes.items()])]
    color_key = kontort.make_color_key(colors[:, ::-1], class_names)
    try:
        os.makedirs('%s/view' % out_root)
    except OSError:
        pass
    cv2.imwrite('%s/view/key.png' % out_root, color_key)



def convert_color(image):
    image = imfeat.convert_image(image, [('opencv', 'lab', cv.IPL_DEPTH_8U)])
    image = np.asarray(cv.GetMat(image)).copy()
    return image


HOG = imfeat.HOGLatent(4)
LBP = imfeat.LBP()
GRAD = imfeat.GradientHistogram()




def convert_labels_to_integrals(label_mask, num_vals):
    masks = []
    print(label_mask.shape)
    for x in range(num_vals):
        m = np.asfarray(label_mask == x)
        m = cv2.integral(m)
        masks.append(m)
    return np.ascontiguousarray(np.swapaxes(np.swapaxes(np.array(masks), 0, 2), 0, 1))


def save_feature_hists(func_data, extra_dump_vars, bins=20):
    mp.ion()
    for var_name, data in zip(extra_dump_vars, zip(*func_data)):
        mp.clf()
        mp.hist(data, bins)
        mp.title(var_name)
        mp.savefig('%s/%s.png' % (out_root, var_name))
        print(var_name)
        

def predict_tree2():
    with open('texton_forest-%s.pkl' % dataset._name) as fp:
        tp = kontort.TextonPredict(pickle.load(fp))
    with open('texton_forest2-%s.pkl' % dataset._name) as fp:
        tp2 = kontort.IntegralPredict(pickle.load(fp))
    save_key()
    save_feature_hists(tp2.func_data, tp2.extra_dump_vars)
    num_masks = tp2.func_data[0][tp2.extra_dump_vars.index('num_masks')]
    print('Func Data2: [%s]' % str(tp2.func_data[0]))
    for num, (masks, image) in enumerate(dataset.segmentation_boxes()):
        ratio, image = resize(image)
        orig_image = image
        image = convert_color_bgr(image)
        image = make_masks(image)
        max_classes1, max_probs1, leaves1, all_probs1 = tp.predict(image, leaves=True, all_probs=True)
        print('Num Tree 1 Leaves[%d]' % tp.num_leaves)
        pred_integrals = convert_leaves_all_probs_pred(image, leaves1, all_probs1, tp.num_leaves)
        preds, integrals = pred_integrals
        print('num_masks[%s] integrals_shape[%s]' % (num_masks, integrals.shape))
        assert num_masks == integrals.shape[-1]
        max_classes2, max_probs2, all_probs2 = tp2.predict(pred_integrals, all_probs=True)
        out = colors[max_classes2]
        del integrals
        try:
            os.makedirs(out_root + '/view')
        except OSError:
            pass
        cv2.imwrite('%s/view/%.05d-mask.png' % (out_root, num), out)
        orig_image.save('%s/view/%.05d-img.png' % (out_root, num))
        cv2.imwrite('%s/view/%.05d-texton.png' % (out_root, num), texton_colors[leaves1])
        with gzip.GzipFile('%s/view/%.05d.pkl.gz' % (out_root, num), 'w') as fp:
            pickle.dump({'max_classes1': max_classes1,
                         'max_probs1': max_probs1,
                         'leaves1': leaves1,
                         'all_probs1': all_probs1,
                         'max_classes2': max_classes2,
                         'max_probs2': max_probs2,
                         'all_probs2': all_probs2,
                         'gt_masks': masks,
                         'image': image,
                         'classes': classes}, fp, -1)


def predict_tree1():
    with open('texton_forest.pkl') as fp:
        tp = kontort.TextonPredict(pickle.load(fp))
    print(tp.num_leaves)
    save_feature_hists(tp.func_data, ['feat_type', 'x0', 'y0', 'b0', 'x1', 'y1', 'b1'])
    rnd = lambda : np.array(np.random.randint(0, 255, 1000), dtype=np.uint8)
    colors = np.ascontiguousarray(np.array([rnd(), rnd(), rnd()]).T)
    try:
        os.makedirs('view')
    except OSError:
        pass
    for num, (masks, image) in enumerate(dataset.segmentation_boxes()):
        #image = resize(image)
        ratio, image = resize(image)
        image_orig = convert_color(image)
        image = make_masks(image_orig)
        print(image.shape)
        a = tp.predict(image, leaves=True)[0]
        cv2.imwrite('view/%.05d-img.png' % num, image_orig)
        cv2.imwrite('view/%.05d-texton.png' % num, texton_colors[a])

if __name__ == '__main__':
    import vision_data
    dataset = vision_data.MSRC()
    out_root = 'texton_data/run-%f' % time.time()
    classes = msrc_classes
    predict_tree2()
