from argparse import Namespace
import argparse


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="I2T")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default=r"/data/bggan/data/xray_h5",
                        help="Directory contains processed dataset.")

    parser.add_argument("--save_dir",
                        type=str,
                        default=r"/data/bggan/data/test_out",
                        help="Directory to save the output files.")

    parser.add_argument("--config_path",
                        type=str,
                        default=r"/data/bggan/code/cfg/config.json",
                        help="Path for the configuration json file.")

    parser.add_argument("--checkpoint_name",
                        type=str,
                        default=r"/data/bggan/code/savemodel/checkpoint_best.pth.tar",
                        help="Path for the configuration json file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu", 
        help='path to pre-trained word Embedding.')

    args = parser.parse_args()

    return args
