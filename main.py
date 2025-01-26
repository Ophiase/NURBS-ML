import argparse
import yaml
from demonstrations import basic, interpolation, surface, synthetic_generation

def main():
    parser = argparse.ArgumentParser(description="NURBS Demonstration System")
    parser.add_argument('--demo', 
        choices=['basic', 'interpolation', 'surface', 'synthetic'],
        help="Demonstration to run")
    parser.add_argument('--train', action='store_true',
                      help="Train the NURBS-ML model")
    parser.add_argument('--config', type=str, default="configs/synthetic_curve.yaml",
                      help="Path to config file")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.train:
        pass
    elif args.demo:
        match args.demo:
            case 'basic': basic.run()
            case 'interpolation': interpolation.run()
            case 'surface': surface.run()
            case 'synthetic': synthetic_generation.run(config)
    else:
        print("Please specify either --train or --demo")

if __name__ == "__main__":
    main()