import hadoopy
import imfeat
import numpy as np
import random
import vision_data
import time
import cv2
import json


radius = 10
msrc_classes = {'building': 9, 'sheep': 11, 'flower': 19, 'bicycle': 7, 'cow': 15, 'face': 13, 'sky': 8, 'tree': 1, 'dog': 4, 'sign': 0, 'water': 10, 'book': 20, 'body': 12, 'cat': 6, 'boat': 14, 'aeroplane': 17, 'car': 18, 'chair': 16, 'grass': 3, 'bird': 2, 'road': 5}
finder_classes = dict((y, x) for x, y in enumerate(['mountain', 'fence', 'car', 'rock', 'tree', 'water', 'sand', 'road', 'house', 'boat', 'grass', 'pier', 'vegetation', 'sky']))
sun397_classes = dict((y, x) for x, y in enumerate(json.load(open('classes.js'))))
GRAD = imfeat.GradientHistogram()


def make_masks(image):
    #hog_masks = HOG.make_feature_mask(image)
    #hog_masks_shape = hog_masks.shape
    #print(hog_masks.shape)
    #hog_masks = hog_masks.reshape((hog_masks.shape[0] * hog_masks.shape[1], hog_masks.shape[2]))
    #hog_masks_min = np.min(hog_masks, 0)
    #hog_masks_max = np.max(hog_masks, 0)
    #print('HoG Min[%s] Max[%s]' % (np.min(hog_masks_min), np.max(hog_masks_max)))
    #print(hog_masks.shape)
    #hog_masks = np.array(255 * np.clip(hog_masks, 0, 1), dtype=np.uint8)
    #hog_masks,
    #LBP.make_feature_mask(image_gray,
    #pool_radius=3),
    image = imfeat.convert_image(image, [{'type': 'numpy', 'mode': 'bgr', 'dtype': 'uint8'}])
    image_gray = imfeat.convert_image(image, [{'type': 'numpy', 'mode': 'gray', 'dtype': 'uint8'}])
    image_gradient = np.array(GRAD.make_feature_mask(np.array(image_gray, dtype=np.float32) / 255.) * 255, dtype=np.uint8)
    image_lab = imfeat.convert_image(image, [{'type': 'numpy', 'mode': 'lab', 'dtype': 'uint8'}])
    return np.ascontiguousarray(np.dstack([image, image_lab, image_gradient]), dtype=np.uint8)


def resize(image, max_side=320):
    height, width = image.shape[:2]
    ratio = 1.
    if max(width, height) <= max_side:
        return ratio, image
    if width > height:
        height = int(max_side * height / float(width))
        ratio = max_side / float(width)
        width = max_side
    else:
        width = int(max_side * width / float(height))
        ratio = max_side / float(height)
        height = max_side
    return ratio, cv2.resize(image, (width, height))


def write_texton_hadoop(dataset, classes):
    """Writes (image_name, image_label_points)

    image_name: A string
    image_label_points: List of (image, [(label, points), ...]) where points is Nx2 (y, x)
    """
    if not isinstance(classes, dict):
        classes = dict((y, x) for x, y in enumerate(classes))
    sample_points = 15000
    samples_per_class = {}

    def make_data():
        for image_num, (masks, image) in enumerate(dataset.segmentation_boxes()):
            ratio, image = resize(image)
            if image.shape[0] < radius * 2 + 1 or image.shape[1] < radius * 2 + 1:
                continue
            image = make_masks(image)
            image_size = float(image.shape[0] * image.shape[1])
            print(image.shape)
            print(image_num)
            label_points = []
            for class_name, mask in masks.items():
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                assert mask.shape == image.shape[:2]
                try:
                    class_num = classes[class_name]
                except KeyError:
                    continue
                ijs = np.dstack(mask.nonzero())[0]
                orig_ijs = ijs
                ijs = np.ascontiguousarray(random.sample(ijs, min(sample_points, len(ijs))))
                if not len(ijs):
                    continue
                print('Image[%d][%s][%d][%d][%f] has ijs available' % (image_num, class_name, len(ijs), len(orig_ijs), len(orig_ijs) / image_size))
                try:
                    samples_per_class[class_name] += len(ijs)
                except KeyError:
                    samples_per_class[class_name] = len(ijs)
                label_points.append((class_num, np.array(ijs, dtype=np.int32)))  # * ratio
            if not label_points:
                print('Image[%d] has no points available' % image_num)
                continue
            print(samples_per_class)
            yield str(image_num), (image, label_points)
    hdfs_file_cnt = 0
    hdfs_buf = []
    start_time = time.time()
    for x in make_data():
        print('spatial_queries/input/%s/%f/%d.tb.seq' % (dataset._name, start_time, hdfs_file_cnt))
        hdfs_buf.append(x)
        if len(hdfs_buf) >= 100:
            try:
                hadoopy.writetb('spatial_queries/input/%s/%f/%d.tb.seq' % (dataset._name, start_time, hdfs_file_cnt), hdfs_buf)
            except IOError, e:
                print('Got IOError, skipping')
                print(e)
            hdfs_file_cnt += 1
            hdfs_buf = []
    if hdfs_buf:
        hadoopy.writetb('spatial_queries/input/%s/%f/%d.tb.seq' % (dataset._name, start_time, hdfs_file_cnt), hdfs_buf)
    print('NumClasses[%d]' % len(classes))
    print('Classes: %r' % classes)

if __name__ == '__main__':
    dataset = vision_data.MSRC()
    classes = msrc_classes
    if 1:
        from data_sources import data_source_from_uri
        from sun397_dataset import SUN397
        uri = 'hbase://localhost:9090/images?image=data:image_320&gt=feat:masks_gt'
        dataset = SUN397(data_source_from_uri(uri))
        classes = json.load(open('classes.js'))
    write_texton_hadoop(dataset, classes)
