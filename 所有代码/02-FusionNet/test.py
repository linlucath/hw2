"""Evaluate the two trained networks, official weights and LMS together."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_pansharpening import main

if __name__ == '__main__':
    main()
