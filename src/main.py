import argparse
from app import App
import os

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--device", help="device", default="cpu")
    parser.add_argument("-vp", "--video_path", help="Video path file or url", default=os.environ.get('video_path'))
    parsed_args: argparse.Namespace = parser.parse_args()
    App().run(parsed_args.video_path)
