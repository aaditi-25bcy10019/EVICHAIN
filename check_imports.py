#!/usr/bin/env python
"""Simple test to check if imports work"""

import sys
print("Step 1: Python loaded")

try:
    import numpy as np
    print("Step 2: numpy imported")
except Exception as e:
    print(f"Failed to import numpy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("Step 3: pandas imported")
except Exception as e:
    print(f"Failed to import pandas: {e}")
    sys.exit(1)

try:
    from sklearn.ensemble import IsolationForest
    print("Step 4: sklearn imported")
except Exception as e:
    print(f"Failed to import sklearn: {e}")
    sys.exit(1)

print("\nAll imports successful! Now running train.py...\n")
