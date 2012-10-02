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
