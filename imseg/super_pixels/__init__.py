import subprocess
import cv2
from . import __path__
import os
import tempfile
import imfeat
import shutil
import Image
import stat


class SuperPixels(object):

    def __init__(self, num_components=50):
        self.num_components = num_components
        self.exec_path = os.path.abspath(__path__[0]) + '/data/superpixels'
        if not os.path.exists(self.exec_path):
            raise ValueError('Cannot find superpixel executable')
        self.temp_dir = tempfile.mkdtemp()
        shutil.copy(self.exec_path, self.temp_dir)
        os.chmod(self.temp_dir + '/superpixels', stat.S_IXUSR | stat.S_IRUSR)
        print(self.temp_dir)

    #def __del__(self):
    #    shutil.rmtree(self.temp_dir)

    @classmethod
    def label_image_to_contours(self, labels, scale=1.):
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

    @classmethod
    def label_image_to_boundaries(cls, labels):
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

    @classmethod
    def label_image_to_colors(cls, labels):
        labels = labels.astype(np.int32)
        labels = labels[:, :, 2] + labels[:, :, 1] * 256 + labels[:, :, 0] * 65536
        num_labels = np.max(labels) + 1
        assert num_labels < 1024
        colors = (np.random.random((num_labels, 3)) * 255).astype(np.uint8)
        return colors[labels.ravel()].reshape((labels.shape[0], labels.shape[1], 3))


    def __call__(self, image, num_components=None):
        if num_components is None:
            num_components = self.num_componenets
        image = imfeat.convert_image(image, {'type': 'numpy', 'dtype': 'uint8', 'mode': 'gray'})
        input_path = self.temp_dir + '/input.pgm'
        output_path = self.temp_dir + '/output.ppm'
        #image = Image.fromarray(image)
        #image.save(input_path)
        cv2.imwrite(input_path, image)
        cur_dir = os.path.abspath('.')
        os.chdir(self.temp_dir)
        cmd = './superpixels input.pgm output.ppm %d' % (num_components,)
        subprocess.call(cmd.split())
        out = cv2.imread(output_path)
        os.chdir(cur_dir)
        return out
