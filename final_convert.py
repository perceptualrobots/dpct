import os
import re
import json
import glob

def convert_vscode_cell_to_jupyter(notebook_path, output_path=None):
    """
    Convert a VS Code Cell format notebook to Jupyter format, replacing gym_name with env_name.
    
    Args:
        notebook_path (str): Path to the notebook in VS Code Cell format
        output_path (str, optional): Path to save the new Jupyter notebook, defaults to same as input
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    if output_path is None:
        output_path = notebook_path
        
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Read the input file
        with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Create backup
        backup_dir = os.path.join(os.path.dirname(notebook_path), 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(notebook_path) + '.final.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Print some debug info
        print(f"Processing: {notebook_path}")
        print(f"  File size: {os.path.getsize(notebook_path)} bytes")
        print(f"  Created backup at: {backup_path}")
        
        # Check if it contains VSCode.Cell tags
        if '<VSCode.Cell' not in content:
            print(f"  No VSCode.Cell tags found in {notebook_path}")
            return False
            
        # Update gym_name to env_name in the content
        content = content.replace('def __init__(self, gym_name,', 'def __init__(self, env_name,')
        content = content.replace('self.gym_name = gym_name', 'self.env_name = env_name')
        content = content.replace('self.env = gym.make(self.gym_name)', 'self.env = gym.make(self.env_name)')
        content = content.replace("'gym_name': self.gym_name,", "'env_name': self.env_name,")
        content = content.replace('gym_name=', 'env_name=')
        content = content.replace('- gym_name:', '- env_name:')
        
        # Extract cells
        pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            print(f"  No cells matched in {notebook_path}")
            return False
            
        print(f"  Found {len(matches)} cells")
        
        # Create Jupyter notebook structure
        jupyter_cells = []
        for cell_id, language, source in matches:
            cell_type = "markdown" if language == "markdown" else "code"
            
            # Process source
            source_lines = source.strip().split('\n')
            
            # Create cell structure
            cell = {
                "cell_type": cell_type,
                "metadata": {
                    "id": cell_id
                },
                "source": source_lines
            }
            
            # Add execution_count and outputs for code cells
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            jupyter_cells.append(cell)
        
        # Create the Jupyter notebook structure
        jupyter_notebook = {
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
        
        # Write the Jupyter notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jupyter_notebook, f, indent=2)
            
        print(f"  Successfully converted to {output_path}")
        print(f"  New file size: {os.path.getsize(output_path)} bytes")
        
        return True
    
    except Exception as e:
        print(f"  Error converting {notebook_path}: {str(e)}")
        return False

def main():
    # Notebook directory
    nbs_dir = "nbs"
    
    # Convert each notebook
    notebooks = glob.glob(os.path.join(nbs_dir, "*.ipynb"))
    print(f"Found {len(notebooks)} notebooks")
    
    success_count = 0
    for notebook in notebooks:
        print(f"\n--- Processing {notebook} ---")
        if convert_vscode_cell_to_jupyter(notebook):
            success_count += 1
    
    print(f"\nConverted {success_count} out of {len(notebooks)} notebooks")
    
    # Try to run nbdev_export
    if success_count > 0:
        print("\nAttempting to run nbdev_export...")
        try:
            import subprocess
            result = subprocess.run(["nbdev_export"], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("nbdev_export completed successfully")
            else:
                print(f"nbdev_export failed with return code {result.returncode}")
                print("Error output:")
                print(result.stderr)
        except Exception as e:
            print(f"Error running nbdev_export: {str(e)}")

if __name__ == "__main__":
    main()
