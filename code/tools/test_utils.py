from argparse import Namespace
import argparse


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/data/coding/bggan/data/xray_h5",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument("--save_dir",
                        type=str,
                        default="/data/coding/bggan/data/test_out/new_0627.1643",
                        help="Directory to save the output files.")

    parser.add_argument("--config_path",
                        type=str,
                        default="/data/coding/bggan/code/cfg/config.json",
                        help="Path for the configuration json file.")

    parser.add_argument("--checkpoint_name",
                        type=str,
                        default="/data/coding/bggan/code/checkpoints/2906.1640/ul_checkpoint_best.pth.tar",
                        help="Path for the configuration json file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu, mgpu
        help='path to pre-trained word Embedding.')

    args = parser.parse_args()

    return args
