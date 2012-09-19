import imseg
import numpy as np
import cv2
try:
    import unittest2 as unittest
except ImportError:
    import unittest

# Cheat Sheet (method/test) <http://docs.python.org/library/unittest.html>
#
# assertEqual(a, b)       a == b   
# assertNotEqual(a, b)    a != b    
# assertTrue(x)     bool(x) is True  
# assertFalse(x)    bool(x) is False  
# assertRaises(exc, fun, *args, **kwds) fun(*args, **kwds) raises exc
# assertAlmostEqual(a, b)  round(a-b, 7) == 0         
# assertNotAlmostEqual(a, b)          round(a-b, 7) != 0
# 
# Python 2.7+ (or using unittest2)
#
# assertIs(a, b)  a is b
# assertIsNot(a, b) a is not b
# assertIsNone(x)   x is None
# assertIsNotNone(x)  x is not None
# assertIn(a, b)      a in b
# assertNotIn(a, b)   a not in b
# assertIsInstance(a, b)    isinstance(a, b)
# assertNotIsInstance(a, b) not isinstance(a, b)
# assertRaisesRegexp(exc, re, fun, *args, **kwds) fun(*args, **kwds) raises exc and the message matches re
# assertGreater(a, b)       a > b
# assertGreaterEqual(a, b)  a >= b
# assertLess(a, b)      a < b
# assertLessEqual(a, b) a <= b
# assertRegexpMatches(s, re) regex.search(s)
# assertNotRegexpMatches(s, re)  not regex.search(s)
# assertItemsEqual(a, b)    sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)      all the key/value pairs in a exist in b


def label_image_to_contours(labels, scale=1.):
    if scale != 1.:
        r = lambda x: int(np.round(x * scale))
        labels = cv2.resize(labels, (r(labels.shape[1]), r(labels.shape[0])), interpolation=cv2.INTER_NEAREST)
    labels = labels.astype(np.int32)
    labels = labels[:, :, 2] + labels[:, :, 1] * 256 + labels[:, :, 0] * 65536
    out = []
    unique_labels = np.unique(labels)
    for region in unique_labels:
        mask = (labels == region).astype(np.uint8)
        points = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[0]
        if points:
            for p in points:
                p = np.asarray(p, dtype=np.int32)
                if scale != 1.:
                    p = (np.round(p / float(scale))).astype(np.int32)
                out.append(p.tolist())
    return out


def label_image_to_boundaries(labels):
    labels = labels.astype(np.int32)
    labels = labels[:, :, 2] + labels[:, :, 1] * 256 + labels[:, :, 0] * 65536
    out = {}
    for region in np.unique(labels):
        points = np.dstack((labels == region).nonzero())
        points = points[0, :, ::-1].astype(np.int32)
        points = np.ascontiguousarray(points.reshape(-1, 1, 2))
        if points.size > 0:
            out.append(cv2.convexHull(points).reshape(-1, 2).tolist())
    return out


def label_image_to_colors(labels):
    labels = labels.astype(np.int32)
    labels = labels[:, :, 2] + labels[:, :, 1] * 256 + labels[:, :, 0] * 65536
    num_labels = np.max(labels) + 1
    assert num_labels < 1024
    colors = (np.random.random((num_labels, 3)) * 255).astype(np.uint8)
    return colors[labels.ravel()].reshape((labels.shape[0], labels.shape[1], 3))


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        a = imseg.SuperPixels()
        lena = cv2.imread('lena.ppm')
        out = a(lena, 75)
        cv2.imwrite('lena_color.png', label_image_to_colors(out))
        hulls = label_image_to_contours(out, .5)
        import json
        fp = open('hulls.js', 'w')
        fp.write('hulls=')
        json.dump(hulls, fp, -1, separators=(',', ':'))
        for hull in hulls:
            hull = np.asarray(hull).astype(np.int32).reshape(1, -1, 2)
            cv2.drawContours(lena, hull, -1, (0, 0, 255))
        cv2.imwrite('lena_convex_seg.png', lena)

if __name__ == '__main__':
    unittest.main()
