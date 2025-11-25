import os
import sys
# Add the root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.general_utils.config import parse_arguments
from modules.tester import Tester
import torch

if __name__ == '__main__':
    """
    Main entry point for the testing script.
    - Parses command-line arguments.
    - Sets up the environment.
    - Initializes and runs the Tester.
    """
    # Parse arguments
    args = parse_arguments()

    # Change the current working directory to the script's directory
    os.chdir(os.path.dirname(__file__))

    # Set CUDA device if specified
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"Using GPU: {args.gpu_id}", flush=True)

    # Initialize and run the tester
    tester = Tester(args=args)
    tester.valid()
    tester.test()

    print("\nTesting finished.", flush=True)
