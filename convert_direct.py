import os
import json
import re
import glob
from pathlib import Path

def convert_vscode_to_jupyter(file_path):
    """
    Convert a VS Code notebook to Jupyter format
    
    Args:
        file_path: Path to the notebook
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Converting {file_path} to Jupyter format...")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"  Read {len(content)} bytes from {file_path}")
            print(f"  File starts with: {content[:20]}")
            
        # Check if it's already in JSON format
        try:
            json.loads(content)
            print(f"  {file_path} is already in JSON format")
            return True
        except json.JSONDecodeError as e:
            print(f"  Not in JSON format: {e}")
            
        # Check if it's in VS Code Cell format
        if "<VSCode.Cell" not in content:
            print(f"  {file_path} is not in VS Code Cell format")
            return False
            
        print(f"  Found VS Code Cell format in {file_path}")
            
        # Extract cells from VS Code format
        cells = []
        cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
        matches = re.findall(cell_pattern, content, re.DOTALL)
        
        for cell_id, cell_language, cell_content in matches:
            if cell_language == "markdown":
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {"id": cell_id},
                    "source": cell_content.strip().split('\n')
                })
            else:
                # Fix env_name/gym_name issues in the code
                cell_content = cell_content.replace("gym_name", "env_name")
                cell_content = re.sub(r'env_name,\s*env_name', 'env_name', cell_content)
                cell_content = re.sub(r'self\.env_name\s*=\s*env_name\s*self\.env_name\s*=\s*env_name', 
                                   'self.env_name = env_name', cell_content)
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"id": cell_id},
                    "outputs": [],
                    "source": cell_content.strip().split('\n')
                })
                
        # Create Jupyter notebook structure
        jupyter_notebook = {
            "cells": cells,
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
            "nbformat_minor": 5
        }
        
        # Backup the original file
        backup_dir = os.path.join(os.path.dirname(file_path), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, os.path.basename(file_path))
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Write the Jupyter notebook
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(jupyter_notebook, f, indent=1)
        
        print(f"  Successfully converted {file_path} to Jupyter format")
        return True
    except Exception as e:
        print(f"  Error converting {file_path}: {e}")
        return False

def main():
    # Get all notebook files
    notebooks_dir = os.path.abspath("nbs")
    notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))
    
    # Skip already processed files
    notebook_files = [f for f in notebook_files 
                      if not os.path.basename(f).startswith("fixed_") 
                      and not os.path.basename(f).startswith("jupyter_")]
    
    print(f"Found {len(notebook_files)} notebooks to process")
    
    # Convert each notebook
    for notebook_file in notebook_files:
        convert_vscode_to_jupyter(notebook_file)

if __name__ == "__main__":
    main()
