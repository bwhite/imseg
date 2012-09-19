import kontort
import cPickle as pickle
import matplotlib.pyplot as mp
import os
import numpy as np


def save_feature_hists(func_data, threshs, extra_dump_vars, output_path, bins=20):
    try:
        os.makedirs(output_path)
    except OSError:
        pass
    mp.ion()
    for var_name, data in zip(extra_dump_vars, zip(*func_data)):
        mp.clf()
        mp.hist(data, bins)
        mp.title(var_name)
        mp.savefig('%s/%s.png' % (output_path, var_name))
        print(var_name)
    # Threshs
    var_name = 't'
    mp.clf()
    mp.hist(threshs, bins)
    mp.title(var_name)
    mp.savefig('%s/%s.png' % (output_path, var_name))
    print(var_name)
        

def main(model_path, model_type, output_path):
    with open(model_path) as fp:
        if model_type == 'depth':
            tp = kontort.DepthPredict(pickle.load(fp))
        elif model_type == 'texton':
            tp = kontort.TextonPredict(pickle.load(fp))
        elif model_type == 'integral':
            tp = kontort.IntegralPredict(pickle.load(fp))
    save_feature_hists(tp.func_data, tp.t, tp.extra_dump_vars, output_path)


def _parse():
    import argparse
    parser = argparse.ArgumentParser(description='Output model info')
    parser.add_argument('path', help='Input model path')
    parser.add_argument('output_path', help='Output path')
    parser.add_argument('type', default='depth', choices=['depth', 'integral', 'texton'], help='Display type')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse()
    main(args.path, args.type, args.output_path)
