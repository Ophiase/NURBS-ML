import argparse
from demonstrations import basic, interpolation

def main():
    parser = argparse.ArgumentParser(description="NURBS Demonstration System")
    parser.add_argument('--demonstration', choices=['basic', 'interpolation'],
                        help="Specify which demonstration to run")

    args = parser.parse_args()

    if args.demonstration == 'basic':
        basic.run()
    elif args.demonstration == 'interpolation':
        interpolation.run()

if __name__ == "__main__":
    main()
