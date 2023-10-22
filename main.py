import argparse
from src.app import App

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='shows output')
    parsed_args: argparse.Namespace = parser.parse_args()
    App()