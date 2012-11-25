import vision_data
import picarus.api
import numpy as np
import json
import imfeat


class SUN397(vision_data.VisionDataset):
    """
    """

    def __init__(self, data_source):
        super(SUN397, self).__init__(name='sun397',
                                     data_urls={},
                                     homepage='',
                                     bibtexs=None,
                                     overview=None)
        self.data_source = data_source

    def segmentation_boxes(self):
        """
        Yields:
            Dataset as specified by 'split'

            Data is in the form of (masks, numpy array), where
            masks is a dict of boolean masks with keys as class names
        """
        classes = json.load(open('classes.js'))
        for row_key, columns in self.data_source.row_column_values(['gt']):
            if not row_key.startswith('sun397train'):
                continue
            columns = dict(columns)
            print(columns.keys())
            print(repr(row_key))
            image = imfeat.image_fromstring(self.data_source.value(row_key, 'image'))
            masks = (255 * picarus.api.np_fromstring(columns['gt'])).astype(np.uint8)
            class_masks = dict((y, np.ascontiguousarray(masks[:, :, x])) for x, y in enumerate(classes))
            yield class_masks, image
