import os
import re
import json
import glob
from pathlib import Path

def convert_vscode_to_jupyter(vscode_content, output_file):
    """
    Convert VS Code notebook content to Jupyter notebook format
    
    Args:
        vscode_content: String content of a VS Code Cell format notebook
        output_file: Path to save the Jupyter notebook
    """
    # Extract cells using regex
    pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
    matches = re.findall(pattern, vscode_content, re.DOTALL)
    
    print(f"Found {len(matches)} cells in VS Code format")
    
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
    
    # Process each cell
    for cell_id, language, source in matches:
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
    
    # Write the Jupyter notebook to file
    print(f"Writing Jupyter notebook with {len(jupyter_notebook['cells'])} cells to {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jupyter_notebook, f, indent=2)
        
        print(f"Successfully converted to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        return True
    except Exception as e:
        print(f"Error writing file: {str(e)}")
        return False

def convert_notebook(input_path, output_path=None):
    """
    Convert a VS Code Cell format notebook to Jupyter format
    
    Args:
        input_path: Path to the VS Code notebook
        output_path: Path for the output Jupyter notebook (default: input_path with 'jupyter_' prefix)
    """
    try:
        # Get notebook content
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            vscode_content = f.read()
        
        # Create output path if not provided
        if output_path is None:
            dir_path = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            output_path = os.path.join(dir_path, f"jupyter_{base_name}")
        
        # Convert and save
        return convert_vscode_to_jupyter(vscode_content, output_path)
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

def main():
    # Get the notebook we want to convert
    input_notebook = "nbs/02_individual.ipynb"
    
    # Convert
    print(f"Converting {input_notebook}...")
    result = convert_notebook(input_notebook)
    print(f"Conversion result: {result}")

if __name__ == "__main__":
    main()
