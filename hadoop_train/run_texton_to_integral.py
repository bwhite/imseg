import hadoopy
import argparse
import os
import glob
import tempfile
import logging
logging.basicConfig(level=logging.DEBUG)
from leaves_to_probs import save_classifiers


def main():
    parser = argparse.ArgumentParser('Convert image data to integral using hadoop')
    parser.add_argument('model_input', help='Model input (.pkl)')
    parser.add_argument('hdfs_input', help='Path to HDFS input')
    parser.add_argument('hdfs_output', help='Path to HDFS output')
    parser.add_argument('integral_type', choices=['spatial', 'argmax', 'argmax_prob', 'all_prob'], help='Type of integrals to make')
    parser.add_argument('--hdfs_classifier_path', help='Path to classifier data')
    parser.add_argument('--classes', help='Classes to use for the classifier', default=None)
    parser.add_argument('--freeze', action='store_true', help='If set, use hadoopy.launch_frozen instead of hadoopy.launch')
    args = parser.parse_args()
    launcher = hadoopy.launch_frozen if args.freeze else hadoopy.launch

    # NOTE: Manage the classifier paths
    files = ['tree_features.py', 'leaves_to_probs.py']
    fp = tempfile.NamedTemporaryFile(suffix='.pkl.gz')
    cmdenvs = []
    if args.hdfs_classifier_path:
        files.append('hog_8_2_clusters.pkl')
        save_classifiers(args.hdfs_classifier_path, args.classes, fp.name)
        files.append(fp.name)
        cmdenvs.append('CLASSIFIERS_FN=%s' % os.path.basename(fp.name))
    if args.integral_type == 'spatial':
        feature_type = 'texton'
    else:
        feature_type = 'depth'
    files.append(args.model_input)
    launcher(args.hdfs_input, args.hdfs_output, 'texton_to_integral.py',
             files=files,
             jobconfs=['mapred.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec',
                       'mapred.output.compression.type=BLOCK',
                       'mapred.output.compress=True',
                       'mapred.compress.map.output=True',
                       'mapred.map.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec',
                       'mapred.child.java.opts=-Xmx768M'],
             cmdenvs=['TREES_SER_FN=%s' % os.path.basename(args.model_input),
                      'FEATURE_TYPE=%s' % feature_type,
                      'INTEGRAL_TYPE=%s' % args.integral_type])

if __name__ == '__main__':
    main()
