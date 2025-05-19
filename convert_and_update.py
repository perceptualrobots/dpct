import os
import re
import json
import glob
from pathlib import Path

def convert_and_update_notebook(notebook_path):
    """
    Convert a VS Code notebook to Jupyter format and update parameters.
    
    Args:
        notebook_path: Path to the notebook file
    """
    print(f"Processing {notebook_path}...")
    
    # Read the notebook content
    try:
        with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Create a backup
        backup_dir = os.path.join(os.path.dirname(notebook_path), 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(notebook_path) + '.conversion.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Created backup at {backup_path}")
        
        # Check if it's in VS Code Cell format
        if '<VSCode.Cell' in content:
            print("File is in VS Code Cell format")
            
            # Update parameters directly in the VS Code Cell content
            # Replace gym_name with env_name in method signature
            content = re.sub(r'def __init__\(\s*self\s*,\s*env_name\s*,\s*gym_name\s*,', r'def __init__(self, env_name,', content)
            
            # Remove the parameter documentation for gym_name
            content = re.sub(r'- gym_name: .*\n', r'', content)
            
            # Remove self.gym_name assignment
            content = re.sub(r'self\.gym_name = gym_name\s*\n', r'', content)
            
            # Update gym.make calls
            content = re.sub(r'self\.env = gym\.make\(self\.gym_name\)', r'self.env = gym.make(self.env_name)', content)
            
            # Update constructor calls
            content = re.sub(r'DHPCTIndividual\(\s*([^,]+)\s*,\s*([^,]+)\s*,', r'DHPCTIndividual(\1,', content)
            
            # Remove gym_name from configuration
            content = re.sub(r"'gym_name': self\.gym_name,\s*\n", r'', content)
            
            # Extract cells using regex
            cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
            matches = re.findall(cell_pattern, content, re.DOTALL)
            
            print(f"Found {len(matches)} cells")
            
            # Convert to Jupyter format
            jupyter_cells = []
            for cell_id, language, source in matches:
                cell_type = "markdown" if language == "markdown" else "code"
                
                # Clean up source
                lines = source.strip().split('\n')
                
                cell = {
                    "cell_type": cell_type,
                    "metadata": {"id": cell_id},
                    "source": lines
                }
                
                if cell_type == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
                
                jupyter_cells.append(cell)
            
            # Create Jupyter notebook
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
            
            # Write the Jupyter notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(jupyter_nb, f, indent=2)
            
            print(f"Converted to Jupyter format and updated parameters in {notebook_path}")
            return True
            
        elif notebook_path.endswith('.ipynb'):
            try:
                print("Attempting to parse as JSON...")
                notebook = json.loads(content)
                
                if "cells" in notebook:
                    print(f"Valid Jupyter notebook with {len(notebook['cells'])} cells")
                    
                    # Update cells
                    for cell in notebook["cells"]:
                        if cell.get("cell_type") == "code":
                            source = cell.get("source", [])
                            new_source = []
                            
                            for i, line in enumerate(source):
                                # Update method signature
                                if 'def __init__(self, env_name, gym_name,' in line:
                                    new_line = line.replace('def __init__(self, env_name, gym_name,', 'def __init__(self, env_name,')
                                    new_source.append(new_line)
                                # Remove gym_name parameter doc
                                elif '- gym_name:' in line:
                                    continue
                                # Remove self.gym_name assignment
                                elif 'self.gym_name = gym_name' in line:
                                    continue
                                # Update gym.make
                                elif 'self.env = gym.make(self.gym_name)' in line:
                                    new_line = line.replace('self.env = gym.make(self.gym_name)', 'self.env = gym.make(self.env_name)')
                                    new_source.append(new_line)
                                # Remove gym_name from config
                                elif "'gym_name': self.gym_name," in line:
                                    continue
                                # Other lines unchanged
                                else:
                                    new_source.append(line)
                            
                            # Update cell source
                            cell["source"] = new_source
                    
                    # Write updated notebook
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        json.dump(notebook, f, indent=2)
                    
                    print(f"Updated parameters in {notebook_path}")
                    return True
                else:
                    print("Not a valid Jupyter notebook")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                return False
        
        else:
            print("Unknown file format")
            return False
            
    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return False

def main():
    # Define the notebook directory
    nbs_dir = "nbs"
    
    # Get all notebook files
    notebook_files = glob.glob(os.path.join(nbs_dir, "*.ipynb"))
    
    print(f"Found {len(notebook_files)} notebooks")
    
    for notebook in notebook_files:
        convert_and_update_notebook(notebook)
    
    print("All notebooks processed")

if __name__ == "__main__":
    main()
