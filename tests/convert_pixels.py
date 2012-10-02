import glob
import random
import os
import json
from test_super_pixels import *


a = imseg.SuperPixels()
for fn in glob.glob('output/*.jpg'):
    print(fn)
    image = cv2.imread(fn)
    out = a(image, 25)
    hulls = label_image_to_contours(out, 1.)
    fp = open('output/%s.js' % os.path.basename(fn), 'w')
    json.dump({'segments': hulls, 'class_prob': {}}, fp, separators=(',', ':'))
    #for hull in hulls[1:]:
    #    hull = np.asarray(hull).astype(np.int32).reshape(1, -1, 2)
    #    cv2.drawContours(image, hull, -1, (0, 0, 255))
    #cv2.imwrite('output_seg/%s_seg.jpg' % os.path.basename(fn), image)
