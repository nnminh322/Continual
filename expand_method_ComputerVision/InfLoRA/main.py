import json
import argparse
from trainer import train
import ipdb

def main():
    cli_args = setup_parser().parse_args()
    param = load_json(cli_args.config)

    args = vars(cli_args)  # Converting argparse Namespace to a dict.
    seed_override = args.pop('seed', None)
    data_path_override = args.pop('data_path', None)

    args.update(param)  # Add parameters from json

    if seed_override is not None:
        args['seed'] = [seed_override]
    if data_path_override is not None:
        args['data_path'] = data_path_override

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the seed list from the config with a single seed.')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override the dataset root from the config.')

    # # optim
    # parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
    return parser


if __name__ == '__main__':
    main()
