import argparse
from app import App

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="device", default="cpu")
    parser.add_argument("-r", "--reload", help="Allow reload", default=False)
    parsed_args: argparse.Namespace = parser.parse_args()
    App().run(parsed_args.reload)
