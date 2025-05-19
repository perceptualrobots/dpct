import os
import re
import json
import sys

# Redirect stdout to a file
log_file = open("detailed_conversion_log.txt", "w")
sys.stdout = log_file

try:
    print("Starting conversion script")
    
    # Input and output files
    input_file = "nbs/02_individual.ipynb"
    output_file = "nbs/jupyter_02_individual.ipynb"
    
    print(f"Converting {input_file} to {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file {input_file} does not exist")
        exit(1)
    
    # Read content of the file
    print(f"Reading file {input_file}")
    with open(input_file, 'rb') as f:
        content_bytes = f.read()
    
    print(f"Read {len(content_bytes)} bytes")
    print(f"First 100 bytes (hex): {content_bytes[:100].hex()}")
    
    # Decode to text
    content = content_bytes.decode('utf-8', errors='replace')
    print(f"Decoded to {len(content)} characters")
    print(f"First 100 chars: {content[:100]}")
    
    # Find VSCode cells
    print("Looking for VSCode.Cell tags")
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
    for i, (cell_id, language, source) in enumerate(vs_code_cells):
        print(f"Processing cell {i+1}, id={cell_id}, language={language}")
        
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
    
    print(f"Processed {len(jupyter_notebook['cells'])} cells")
    
    # Write output
    print(f"Writing to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(jupyter_notebook, f, indent=2)
    
    print(f"Successfully wrote {output_file} ({os.path.getsize(output_file)} bytes)")
    
    # Try to run nbdev_export with the new notebook
    print("\nNow trying to export with nbdev")
    try:
        import subprocess
        cmd = f"nbdev_export --path {output_file}"
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Exit code: {result.returncode}")
        print(f"Standard output: {result.stdout}")
        print(f"Standard error: {result.stderr}")
    except Exception as e:
        print(f"Error running nbdev_export: {str(e)}")

except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Close the log file and restore stdout
    sys.stdout = sys.__stdout__
    log_file.close()
    print("Conversion completed. See detailed_conversion_log.txt for details.")
