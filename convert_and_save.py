import os
import re
import json

try:
    # Input and output files
    input_file = "nbs/02_individual.ipynb"
    output_file = "nbs/jupyter_02_individual.ipynb"
    
    print(f"Starting conversion from {input_file} to {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} does not exist")
        exit(1)
    
    # Read content
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print(f"Read {len(content)} bytes from {input_file}")
    
    # Check for VSCode format
    vs_code_cells = re.findall(r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>', content, re.DOTALL)
    
    print(f"Found {len(vs_code_cells)} VSCode cells")
    
    if len(vs_code_cells) == 0:
        print("No VSCode cells found. Exiting.")
        exit(1)
    
    # Create Jupyter notebook structure
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
    
    # Convert each cell
    for cell_id, language, source in vs_code_cells:
        cell_type = "markdown" if language == "markdown" else "code"
        
        source_lines = source.strip().split('\n')
        
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source_lines
        }
        
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        
        jupyter_notebook["cells"].append(cell)
    
    print(f"Converted {len(jupyter_notebook['cells'])} cells")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jupyter_notebook, f, indent=2)
    
    print(f"Wrote {output_file} successfully")
    
    # Print some stats
    print("\nConversion summary:")
    print(f"Input file: {input_file}, {len(content)} bytes")
    print(f"Output file: {output_file}, {os.path.getsize(output_file)} bytes")
    print(f"Cells: {len(jupyter_notebook['cells'])}")
    cell_types = {}
    for cell in jupyter_notebook["cells"]:
        cell_type = cell["cell_type"]
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
    for cell_type, count in cell_types.items():
        print(f"  {cell_type}: {count}")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
