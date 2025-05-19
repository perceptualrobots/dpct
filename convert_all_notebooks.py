import os
import glob
import re
import json
import uuid
from pathlib import Path

def convert_notebook(input_file):
    """Convert a VS Code notebook to Jupyter format."""
    print(f"Processing: {input_file}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Create backup
    backup_dir = os.path.join(os.path.dirname(input_file), "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_file = os.path.join(backup_dir, os.path.basename(input_file) + '.backup')
    
    if not os.path.exists(backup_file):
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Created backup at {backup_file}")
    
    # Check if it contains VSCode.Cell tags
    if '<VSCode.Cell' not in content:
        print(f"  No VSCode.Cell tags found in {input_file}")
        return
    
    # Extract cells using regex
    cells = []
    cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
    matches = re.findall(cell_pattern, content, re.DOTALL)
    
    print(f"  Found {len(matches)} cells")
    
    # Convert cells to Jupyter format
    jupyter_cells = []
    for cell_id, language, source in matches:
        cell_type = "markdown" if language == "markdown" else "code"
        
        # Clean up source code
        source_lines = source.strip().split('\n')
        
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines
        }
        
        # Add execution_count and outputs for code cells
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        
        jupyter_cells.append(cell)
    
    # Create Jupyter notebook structure
    jupyter_nb = {
        "cells": jupyter_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the Jupyter notebook to file
    output_file = input_file
    print(f"  Writing converted notebook to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jupyter_nb, f, indent=2)

def main():
    notebook_dir = 'nbs'
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks to convert")
    
    for notebook in notebooks:
        convert_notebook(notebook)
        print(f"  Converted {notebook}")
    
    print("All notebooks converted")

if __name__ == "__main__":
    main()
