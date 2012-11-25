import argparse
import json
import tempfile
import subprocess
import glob
import time


TEXTON_JS = '{"input_path": ["INPUT_PATH"], "max_reducers": 50, "num_thresh": 100, "min_info": 0.0, "max_levels": 6, "num_feat": 1000, "seed": 1234, "num_classes": NUM_CLASSES, "feature_type": "texton", "output_path": "/user/brandyn/spatial_queries/output/texton", "radius": 10, "max_root_buffer": 8}'

INTEGRAL_JS = '{"input_path": ["INPUT_PATH"], "max_reducers": 50, "num_thresh": 100, "min_info": 0.01, "max_levels": 8, "num_feat": 1000, "num_classes": NUM_CLASSES, "feature_type": "integral", "output_path": "/user/brandyn/spatial_queries/output/integral", "num_integrals": NUM_INTEGRALS, "num_ilp_dims": 0, "ilp_prob": 0.0, "max_box_dist_radius_y": 40, "max_box_dist_radius_x": 40, "min_box_radius_y": 5, "min_box_radius_x": 5, "max_box_radius_y": 30, "max_box_radius_x": 30, "max_root_buffer": 8, "min_sparsity": 0.1}'


def call(cmd):
    print(cmd)
    subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('hdfs_input',
                        help='HDFS Input')
    parser.add_argument('num_classes',
                        help='Num Classes', type=int)
    parser.add_argument('--test_input',
                        help='Test input dir')
    args = parser.parse_args()
    temp_dir = tempfile.mkdtemp()
    texton_input_fn = temp_dir + '/texton.js'
    integral_input_fn = temp_dir + '/integral.js'
    run_time = str(time.time())
    output_dir = 'job_output/%s' % run_time
    open(texton_input_fn, 'w').write(TEXTON_JS.replace('INPUT_PATH', args.hdfs_input).replace('NUM_CLASSES', str(args.num_classes)))
    call(('python driver.py %s %s' % (output_dir, texton_input_fn)).split())
    texton_model_fn = glob.glob(output_dir + '/*5.pkl')[0]
    integral_path = '/user/brandyn/spatial_queries/output/integral/%s' % run_time
    call(('python run_texton_to_integral.py %s %s %s spatial' % (texton_model_fn, args.hdfs_input, integral_path)).split())
    open(integral_input_fn, 'w').write(INTEGRAL_JS.replace('INPUT_PATH', integral_path).replace('NUM_CLASSES', str(args.num_classes)).replace('NUM_INTEGRALS', str(args.num_classes + 64)))
    call(('python driver_multi.py 3 %s %s' % (output_dir, integral_input_fn)).split())
    call(('python merge_models.py %s/tree_ser-integral.pkl %s' % (output_dir, ' '.join(glob.glob(output_dir + '/*7.pkl')))).split())
    # TODO Predict model on input data (optional)


if __name__ == '__main__':
    args = main()
