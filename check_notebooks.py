import os
import json
import glob
from pathlib import Path

def check_notebook(notebook_path):
    """
    Check if a notebook is a valid JSON file and fix it if needed.
    
    Args:
        notebook_path: Path to the notebook file
    
    Returns:
        bool: True if the notebook is valid, False otherwise
    """
    print(f"Checking {notebook_path}...")
    
    try:
        # Check if file exists and has content
        if not os.path.exists(notebook_path):
            print(f"  Error: File does not exist")
            return False
        
        file_size = os.path.getsize(notebook_path)
        print(f"  File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"  Error: File is empty")
            return False
        
        # Read file content
        with open(notebook_path, 'rb') as f:
            content = f.read(100)
            print(f"  First 100 bytes (hex): {content.hex()[:100]}")
            print(f"  First 100 bytes (ascii): {repr(content[:100])}")
        
        # Try to parse as JSON
        with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Check for VS Code Cell format
        if '<VSCode.Cell' in content:
            print(f"  Warning: File contains VSCode.Cell tags")
            
            # Create a backup
            backup_dir = os.path.join(os.path.dirname(notebook_path), "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(notebook_path) + '.backup')
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created backup at {backup_path}")
            
            # Extract cells
            import re
            pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
            matches = re.findall(pattern, content, re.DOTALL)
            
            print(f"  Found {len(matches)} cells")
            
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
            
            print(f"  Fixed VS Code Cell format in {notebook_path}")
            return True
        
        try:
            # Try parsing as JSON
            notebook = json.loads(content)
            
            # Check if it's a valid Jupyter notebook
            if not isinstance(notebook, dict) or "cells" not in notebook:
                print(f"  Error: Not a valid Jupyter notebook structure")
                return False
            
            # Check cells
            for i, cell in enumerate(notebook["cells"]):
                if not isinstance(cell, dict) or "source" not in cell:
                    print(f"  Error: Cell {i} is not valid")
                    return False
            
            print(f"  Valid Jupyter notebook with {len(notebook['cells'])} cells")
            return True
            
        except json.JSONDecodeError as e:
            print(f"  Error: Invalid JSON: {str(e)}")
            
            # Try fixing common issues
            if content.startswith('<!-- '):
                # VS Code comment at the beginning
                content = '{\n' + content.split('\n', 1)[1]
                
                try:
                    # Try parsing again
                    notebook = json.loads(content)
                    
                    # Write fixed content
                    with open(notebook_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  Fixed VS Code comment issue in {notebook_path}")
                    return True
                except:
                    pass
            
            return False
            
    except Exception as e:
        print(f"  Error checking notebook: {str(e)}")
        return False

def main():
    notebook_dir = "nbs"
    notebooks = glob.glob(os.path.join(notebook_dir, "*.ipynb"))
    
    output_file = "notebook_check_results.txt"
    with open(output_file, 'w', encoding='utf-8') as log:
        log.write(f"Found {len(notebooks)} notebooks\n")
        print(f"Found {len(notebooks)} notebooks")
        
        for notebook in notebooks:
            log.write(f"\nChecking {notebook}...\n")
            print(f"Checking {notebook}...")
            
            # Check file exists and has content
            if not os.path.exists(notebook):
                log.write(f"  Error: File does not exist\n")
                print(f"  Error: File does not exist")
                continue
            
            file_size = os.path.getsize(notebook)
            log.write(f"  File size: {file_size} bytes\n")
            print(f"  File size: {file_size} bytes")
            
            if file_size == 0:
                log.write(f"  Error: File is empty\n")
                print(f"  Error: File is empty")
                continue
            
            # Try to read file content
            try:
                with open(notebook, 'r', encoding='utf-8', errors='replace') as f:
                    first_line = f.readline().strip()
                    log.write(f"  First line: {first_line[:100]}\n")
                    print(f"  First line: {first_line[:100]}")
                    
                    if first_line.startswith('<!-- filepath:'):
                        log.write("  Warning: File starts with VS Code comment\n")
                        print("  Warning: File starts with VS Code comment")
                        
                        # Read the rest of the file
                        f.seek(0)
                        content = f.read()
                        
                        # Check for VS Code Cell format
                        if '<VSCode.Cell' in content:
                            log.write("  File is in VS Code Cell format, converting...\n")
                            print("  File is in VS Code Cell format, converting...")
                            
                            # Create a backup
                            backup_dir = os.path.join(os.path.dirname(notebook), "backup")
                            os.makedirs(backup_dir, exist_ok=True)
                            backup_path = os.path.join(backup_dir, os.path.basename(notebook) + '.backup')
                            
                            with open(backup_path, 'w', encoding='utf-8') as bf:
                                bf.write(content)
                            log.write(f"  Created backup at {backup_path}\n")
                            print(f"  Created backup at {backup_path}")
                            
                            # Extract cells
                            import re
                            pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
                            matches = re.findall(pattern, content, re.DOTALL)
                            
                            log.write(f"  Found {len(matches)} cells\n")
                            print(f"  Found {len(matches)} cells")
                            
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
                            with open(notebook, 'w', encoding='utf-8') as f:
                                json.dump(jupyter_nb, f, indent=2)
                            
                            log.write(f"  Converted VS Code Cell format in {notebook}\n")
                            print(f"  Converted VS Code Cell format in {notebook}")
                    elif first_line.startswith('{'):
                        log.write("  File appears to be in JSON format\n")
                        print("  File appears to be in JSON format")
                        
                        # Try parsing as JSON
                        try:
                            with open(notebook, 'r', encoding='utf-8') as f:
                                notebook_json = json.load(f)
                            
                            # Check if it's a valid Jupyter notebook
                            if "cells" in notebook_json:
                                log.write(f"  Valid Jupyter notebook with {len(notebook_json['cells'])} cells\n")
                                print(f"  Valid Jupyter notebook with {len(notebook_json['cells'])} cells")
                            else:
                                log.write("  JSON doesn't appear to be a Jupyter notebook\n")
                                print("  JSON doesn't appear to be a Jupyter notebook")
                        except json.JSONDecodeError as e:
                            log.write(f"  Error parsing JSON: {str(e)}\n")
                            print(f"  Error parsing JSON: {str(e)}")
                    else:
                        log.write(f"  Unknown file format: {first_line[:50]}\n")
                        print(f"  Unknown file format: {first_line[:50]}")
            except Exception as e:
                log.write(f"  Error reading file: {str(e)}\n")
                print(f"  Error reading file: {str(e)}")
        
        log.write("\nAll notebooks checked\n")
        print("\nAll notebooks checked")

if __name__ == "__main__":
    main()
