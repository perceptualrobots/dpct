import os
import re
import json
import traceback

# Setup logging to a file
log_file = "conversion_with_detailed_log.txt"
with open(log_file, 'w') as f:
    f.write("Starting conversion process\n")

def log(message):
    """Write a message to the log file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + "\n")

try:
    # Define the notebook path
    notebook_path = r"c:\Users\ruper\Versioning\python\nbdev\dpct\nbs\02_individual.ipynb"
    output_path = r"c:\Users\ruper\Versioning\python\nbdev\dpct\nbs\jupyter_02_individual_final.ipynb"
    
    log(f"Processing notebook: {notebook_path}")
    
    # Check if file exists
    if not os.path.exists(notebook_path):
        log(f"ERROR: File does not exist: {notebook_path}")
        exit(1)
    
    # Get file size
    file_size = os.path.getsize(notebook_path)
    log(f"File size: {file_size} bytes")
    
    # Read the notebook content
    with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    log(f"Successfully read file, content length: {len(content)}")
    log(f"First 100 characters: {repr(content[:100])}")
    
    # Check if it contains VSCode.Cell tags
    if '<VSCode.Cell' not in content:
        log(f"No VSCode.Cell tags found, exiting")
        exit(1)
    
    log("Found VSCode.Cell tags, proceeding with conversion")
    
    # Update gym_name to env_name
    content = content.replace('def __init__(self, gym_name,', 'def __init__(self, env_name,')
    content = content.replace('self.gym_name = gym_name', 'self.env_name = env_name')
    content = content.replace('self.env = gym.make(self.gym_name)', 'self.env = gym.make(self.env_name)')
    content = content.replace("'gym_name': self.gym_name,", "'env_name': self.env_name,")
    content = content.replace('gym_name=', 'env_name=')
    content = content.replace('- gym_name:', '- env_name:')
    
    log("Updated gym_name references to env_name")
    
    # Extract cells using regex
    pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">(.*?)</VSCode\.Cell>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    log(f"Found {len(matches)} cells in the notebook")
    
    # Create Jupyter notebook structure
    jupyter_cells = []
    for i, (cell_id, language, source) in enumerate(matches):
        cell_type = "markdown" if language == "markdown" else "code"
        
        # Process source
        source_lines = source.strip().split('\n')
        
        # Create cell
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
        
        # Log the first few cells for debugging
        if i < 3:
            log(f"Cell {i+1} - Type: {cell_type}, ID: {cell_id}")
            log(f"  First line of source: {source_lines[0] if source_lines else '(empty)'}")
    
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
    
    log("Created Jupyter notebook structure")
    
    # Write the Jupyter notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(jupyter_notebook, f, indent=2)
    
    log(f"Successfully wrote Jupyter notebook to {output_path}")
    log(f"New file size: {os.path.getsize(output_path)} bytes")
    
    # Verify the new file
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        log(f"Successfully verified the new notebook format with {len(notebook['cells'])} cells")
    except Exception as e:
        log(f"Error verifying the new notebook: {str(e)}")
    
except Exception as e:
    log(f"ERROR: {str(e)}")
    log(traceback.format_exc())
