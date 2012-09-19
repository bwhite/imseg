import hadoopy
import time
import numpy as np
import cPickle as pickle
import tempfile
import os
import json
import argparse
import pprint


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('out_dir',
                        help='Output dir')
    parser.add_argument('json_configs', type=str, nargs='+',
                        help='JSON Configs to load, they are applied in order with the last taking precedence')
    args = parser.parse_args()
    c = {}
    for json_config_fn in args.json_configs:
        c.update(json.load(open(json_config_fn)))
    try:
        np.random.seed(c['seed'])
        del c['seed']
    except KeyError:
        pass
    return args.out_dir, c, [int(np.random.randint(0, 2**31 - 1)) for level in range(c['max_levels'] + 1)]


def main(out_dir, c, level_seeds):
    pprint.pprint(c)  # Output config
    start_time = time.time()
    output_path = '%s/%f/' % (c['output_path'], start_time)
    tree_ser = [[]]
    tree_map = {0: tree_ser}
    frozen_tar_path = None
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    with open('%s/tree_ser-%f.js' % (out_dir, start_time), 'w')  as fp:
        json.dump(c, fp)
    for level in range(c['max_levels'] + 1):  # One more level for leaves
        cur_output_path = '%s/%d/' % (output_path, level)
        cmdenvs = ['%s=%s' % (x.upper(), y) for x, y in c.items()]
        cmdenvs += ['SEED=%d' % level_seeds[level],
                   'LEVEL=%d' % level]
        # This task is run with compression and a modified partitioner
        with tempfile.NamedTemporaryFile() as fp:
            pickle.dump(tree_ser, fp, -1)
            fp.flush()
            cur_cmdenvs = list(cmdenvs)  # Make a copy
            if tree_ser[0]:  # Only use if we have a tree already, if not everything is root=0
                cur_cmdenvs += ['TREE_SER_FN=%s' % os.path.basename(fp.name)]
            frozen_tar_path = hadoopy.launch_frozen(c['input_path'], cur_output_path + 'feat', 'tree_level.py', cmdenvs=cur_cmdenvs,
                                                    num_reducers=min(c['max_reducers'], 2**level),
                                                    files=[fp.name],
                                                    jobconfs=['mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                                              'mapred.output.compression.type=BLOCK',
                                                              'mapred.task.timeout=6000000',
                                                              'mapred.child.java.opts=-Xmx768M'],
                                                    frozen_tar_path=frozen_tar_path)['frozen_tar_path']
        # ,'mapred.child.java.opts=-Xmx512M'
        # Collect output and add to tree, for terminated nodes compute final probabilities
        new_children = False
        for root, (info_gain, ql, qr, feat_ser, num_image) in hadoopy.readtb(cur_output_path + 'feat'):
            print('InfoGain[%f] MinInfo[%f] Level[%d] MaxLevels[%d]' % (info_gain, c['min_info'], level, c['max_levels']))
            if info_gain >= c['min_info'] and level != c['max_levels']:
                new_children = True
                left_tree, right_tree = [ql / float(np.sum(ql))], [qr / float(np.sum(qr))]
                tree_map[root * 2 + 1] = left_tree
                tree_map[root * 2 + 2] = right_tree
                del tree_map[root][0]  # Erase the probabilities we had there, replace with subtree
                tree_map[root] += [feat_ser, left_tree, right_tree, {'info_gain': info_gain}]
        if not new_children:  # We are done
            break
        with open('%s/tree_ser-%s-%f-%d.pkl' % (out_dir, c['feature_type'], start_time, level), 'w')  as fp:
            pickle.dump([tree_ser], fp, -1)

if __name__ == '__main__':
    main(*parse())
