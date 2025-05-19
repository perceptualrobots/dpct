import os
import glob
import json
import re
import uuid

def convert_vscode_to_jupyter(input_file, output_file=None):
    """Convert a VSCode format notebook to Jupyter format."""
    if output_file is None:
        output_file = input_file
    
    print(f"Processing: {input_file}")
    try:
        # Read file as text
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Check if it contains VSCode.Cell tags
        if '<VSCode.Cell' not in content:
            print(f"  No VSCode.Cell tags found in {input_file}")
            return False
            
        # Extract cells using regex
        cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
        cells = re.findall(cell_pattern, content, re.DOTALL)
        
        print(f"  Found {len(cells)} cells")
        
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
        
        # Process cells
        for i, (cell_id, language, source) in enumerate(cells):
            # Determine cell type
            cell_type = "markdown" if language == "markdown" else "code"
            
            # Process source lines
            source_lines = source.strip().split('\n')
            # Wrap in proper format for Jupyter
            source_lines = [line for line in source_lines]
            
            # Create cell dict
            cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": source_lines
            }
            
            # Add outputs and execution_count for code cells
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            jupyter_nb["cells"].append(cell)
        
        # Backup the original file
        backup_dir = os.path.join(os.path.dirname(input_file), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, os.path.basename(input_file))
        
        # Only make a backup if one doesn't already exist
        if not os.path.exists(backup_file):
            print(f"  Creating backup at: {backup_file}")
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Write the new Jupyter notebook
        print(f"  Writing converted notebook to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jupyter_nb, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    notebook_dir = 'nbs'
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))
    
    print(f"Found {len(notebooks)} notebooks to convert")
    
    success_count = 0
    for notebook in notebooks:
        if convert_vscode_to_jupyter(notebook):
            success_count += 1
            print(f"  Converted {notebook} successfully")
        else:
            print(f"  Failed to convert {notebook}")
    
    print(f"\nConversion complete: {success_count}/{len(notebooks)} notebooks converted successfully")

if __name__ == "__main__":
    main()
