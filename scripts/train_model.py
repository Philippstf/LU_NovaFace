#!/usr/bin/env python3
"""
Script to start model training.
"""

import sys
import os
from pathlib import Path

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.trainer import main

if __name__ == "__main__":
    main()