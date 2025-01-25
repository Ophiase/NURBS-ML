import argparse
from demonstrations import basic, interpolation, surface


def main():
    parser = argparse.ArgumentParser(description="NURBS Demonstration System")
    parser.add_argument('--demonstration', choices=['surface', 'basic', 'interpolation'],
                        help="Specify which demonstration to run")

    args = parser.parse_args()

    match args.demonstration:
        case 'basic': basic.run()
        case 'interpolation': interpolation.run()
        case 'surface': surface.run()


if __name__ == "__main__":
    main()
