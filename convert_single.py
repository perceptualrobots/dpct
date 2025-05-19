import re
import json

# Define input and output files
input_file = 'nbs/02_individual.ipynb'
output_file = 'nbs/converted_02_individual.ipynb'

# Read the input file
print(f"Reading {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")
print(f"First 100 chars: {content[:100]}")

# Check for VS Code format
if '<VSCode.Cell' in content:
    print("Found VS Code cell format")
    
    # Extract cells using regex
    pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"Extracted {len(matches)} cells")
    
    # Convert to Jupyter notebook format
    jupyter_notebook = {
        "cells": [],
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
    
    # Process each cell
    for idx, (cell_id, language, source) in enumerate(matches):
        # Determine cell type
        cell_type = "markdown" if language == "markdown" else "code"
        
        # Create cell dictionary
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.strip().split('\n')
        }
        
        # Add outputs for code cells
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        
        # Add cell to notebook
        jupyter_notebook["cells"].append(cell)
        
        # Print some information about the first few cells
        if idx < 2:
            print(f"Cell {idx+1}: Type = {cell_type}, Length = {len(source)} chars")
    
    # Write the Jupyter notebook to file
    print(f"Writing {len(jupyter_notebook['cells'])} cells to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jupyter_notebook, f, indent=2)
    
    print("Conversion complete")
else:
    print("Not in VS Code cell format")
