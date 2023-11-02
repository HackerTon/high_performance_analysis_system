import argparse
from app import App
import os

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="device", default=os.environ.get('device'))
    parser.add_argument("-vp", "--video_path", help="Video path file or url", default=os.environ.get('video_path'))
    parser.add_argument("-b", "--batch_size", help="Batch size", default=os.environ.get('batch_size'), type=int)
    parsed_args: argparse.Namespace = parser.parse_args()

    print(f'Run on {parsed_args.device}')
    print(f'with video of {parsed_args.video_path}')
    print(f'and batch size of {parsed_args.batch_size}')

    App().run(parsed_args.device, parsed_args.video_path, parsed_args.batch_size)
