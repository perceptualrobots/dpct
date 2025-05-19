import os
import json
import re
import glob
import uuid

def inspect_file_content(file_path):
    """Inspect the first few bytes of a file to determine its format."""
    with open(file_path, 'rb') as f:
        # Read first 100 bytes
        header = f.read(100)
        print(f"File: {file_path}")
        print(f"First 100 bytes (hex): {header.hex()}")
        print(f"First 100 bytes (ascii): {repr(header)}")
        
        # Check if it starts with XML or JSON indicators
        if header.startswith(b'<'):
            print("Appears to be XML/HTML format")
            return "xml"
        elif header.startswith(b'{'):
            print("Appears to be JSON format")
            return "json"
        else:
            print("Unknown format")
            return "unknown"

def convert_vscode_to_jupyter(file_path, output_path=None):
    """Convert a VS Code notebook to Jupyter format."""
    if output_path is None:
        output_path = file_path
        
    format_type = inspect_file_content(file_path)
    
    with open(file_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    
    # Check for VSCode.Cell tags
    if '<VSCode.Cell' in content:
        print("Found VSCode.Cell tags")
        
        # Extract all cells using regex
        cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
        cells = re.findall(cell_pattern, content, re.DOTALL)
        
        print(f"Extracted {len(cells)} cells")
        
        # Create a Jupyter notebook structure
        jupyter_nb = {
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
        
        # Convert each VSCode cell to Jupyter cell
        for cell_id, language, source in cells:
            cell_type = "markdown" if language == "markdown" else "code"
            
            # Create cell dict
            cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": source.strip().split('\n'),
                "id": str(uuid.uuid4())
            }
            
            # Add outputs and execution_count for code cells
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            jupyter_nb["cells"].append(cell)
        
        # Create backup directory if not exists
        backup_dir = os.path.join(os.path.dirname(file_path), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup file
        backup_file = os.path.join(backup_dir, os.path.basename(file_path))
        print(f"Creating backup at: {backup_file}")
        with open(backup_file, 'wb') as f:
            f.write(content.encode('utf-8'))
        
        # Write Jupyter notebook
        print(f"Writing converted notebook to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jupyter_nb, f, indent=1)
        
        return True
    else:
        print("No VSCode.Cell tags found. File might already be in Jupyter format.")
        return False

def main():
    notebook_dir = 'nbs'
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks")
    
    for notebook in notebooks:
        print(f"\nProcessing: {notebook}")
        convert_vscode_to_jupyter(notebook)

if __name__ == "__main__":
    main()
