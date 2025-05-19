import os
import glob
import json
import re
import uuid

def get_cell_type(language):
    """Determine the cell type based on the language."""
    if language == "markdown":
        return "markdown"
    return "code"

def convert_vscode_to_jupyter(input_file, output_file=None):
    """Convert a VSCode format notebook to Jupyter format."""
    if output_file is None:
        output_file = input_file
    
    print(f"Reading file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all VSCode cells using regex
        cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
        cells = re.findall(cell_pattern, content, re.DOTALL)
        
        print(f"Found {len(cells)} cells in {input_file}")
        
        if len(cells) == 0:
            print("WARNING: No cells found! Content sample:")
            print(content[:200] + "...")
            return None
        
        # Create Jupyter notebook structure
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
        for i, (cell_id, language, source) in enumerate(cells):
            cell_type = get_cell_type(language)
            print(f"  Cell {i+1}: {cell_type} ({len(source)} chars)")
            
            # Split source by lines and make sure it's a list of strings
            source_lines = source.strip().split('\n')
            
            # Create cell dict
            cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": source_lines,
                "id": str(uuid.uuid4())
            }
            
            # Add outputs and execution_count for code cells
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            jupyter_nb["cells"].append(cell)
        
        # Create backup
        backup_dir = os.path.join(os.path.dirname(input_file), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, os.path.basename(input_file))
        
        print(f"Creating backup at: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Write the Jupyter notebook to file
        print(f"Writing converted notebook to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jupyter_nb, f, indent=1)
        
        return output_file
    except Exception as e:
        print(f"ERROR converting {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Define the directory containing notebooks
    notebook_dir = 'nbs'
    
    # Get list of all notebook files
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks to convert")
    
    success_count = 0
    for notebook in notebooks:
        print(f"\nProcessing {notebook}...")
        result = convert_vscode_to_jupyter(notebook)
        if result:
            success_count += 1
            print(f"Converted {notebook} successfully")
        else:
            print(f"Failed to convert {notebook}")
    
    print(f"\nConversion complete: {success_count}/{len(notebooks)} notebooks converted successfully")

if __name__ == "__main__":
    main()
