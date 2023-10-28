import argparse
from app import App

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='device', default='cpu')
    parsed_args: argparse.Namespace = parser.parse_args()
    App(parsed_args.device).run()