import os
import sys
import glob
from pathlib import Path
import json

# Add the .venv/site-packages to the path to use nbdev modules
sys.path.append("C:/Users/ruper/Versioning/python/nbdev/dpct/.venv/lib/site-packages")

try:
    from execnb.nbio import read_nb
    print("Successfully imported nbdev modules")
except ImportError as e:
    print(f"Failed to import nbdev modules: {e}")
    sys.exit(1)

def check_notebook_nbdev(notebook_path):
    """Check if a notebook can be read by nbdev"""
    print(f"Checking {notebook_path} with nbdev...")
    
    try:
        # Try to read the notebook using nbdev's read_nb function
        nb = read_nb(notebook_path)
        print(f"  Successfully read with nbdev")
        return True
    except Exception as e:
        print(f"  Failed to read with nbdev: {e}")
        return False

def main():
    # Find all notebook files
    notebooks_dir = "C:/Users/ruper/Versioning/python/nbdev/dpct/nbs"
    notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))
    
    valid_count = 0
    invalid_count = 0
    
    for notebook_file in notebook_files:
        if check_notebook_nbdev(notebook_file):
            valid_count += 1
        else:
            invalid_count += 1
    
    print(f"\nResults: {valid_count} valid, {invalid_count} invalid")

if __name__ == "__main__":
    main()
