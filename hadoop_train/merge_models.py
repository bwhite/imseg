import cPickle as pickle


def _parse():
    import argparse
    parser = argparse.ArgumentParser('Merge multiple tree models together')
    parser.add_argument('output_model', help='Path to the output model')
    parser.add_argument('input_models', nargs='+', help='Path to the input models')
    return parser.parse_args()


def main(output_model, input_models):
    trees_ser = []
    for input_model in input_models:
        with open(input_model) as fp:
            trees_ser += pickle.load(fp)
    with open(output_model, 'w') as fp:
        pickle.dump(trees_ser, fp, -1)


if __name__ == '__main__':
    args = _parse()
    main(args.output_model, args.input_models)
