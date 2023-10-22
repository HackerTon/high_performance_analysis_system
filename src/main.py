import argparse
from app import App

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='shows output')
    parsed_args: argparse.Namespace = parser.parse_args()
    App().run_train()