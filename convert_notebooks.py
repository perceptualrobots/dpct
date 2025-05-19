import os
import json
import glob
import nbformat

def convert_vscode_to_jupyter(file_path):
    """Convert a VS Code notebook to Jupyter format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a new Jupyter notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = []
        
        # Parse VS Code cells
        vscode_cells = []
        current_cell = ""
        in_cell = False
        cell_properties = {}
        
        for line in content.split('\n'):
            if "<VSCode.Cell" in line:
                in_cell = True
                # Extract cell properties
                import re
                cell_id = re.search('id="([^"]*)"', line)
                cell_language = re.search('language="([^"]*)"', line)
                
                cell_properties = {
                    'id': cell_id.group(1) if cell_id else None,
                    'language': cell_language.group(1) if cell_language else 'python'
                }
                current_cell = ""
            elif "</VSCode.Cell>" in line:
                in_cell = False
                vscode_cells.append((cell_properties, current_cell))
            elif in_cell:
                current_cell += line + "\n"
        
        # Convert to Jupyter cells
        for props, cell_content in vscode_cells:
            if props['language'] == 'markdown':
                cell = nbformat.v4.new_markdown_cell(cell_content)
            else:
                cell = nbformat.v4.new_code_cell(cell_content)
            
            # Add metadata
            if props['id']:
                if 'metadata' not in cell:
                    cell['metadata'] = {}
                cell['metadata']['id'] = props['id']
            
            nb.cells.append(cell)
        
        # Write the Jupyter notebook
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        return False

# Convert all notebooks in the nbs directory
notebooks = glob.glob('nbs/*.ipynb')
for notebook in notebooks:
    print(f"Converting {notebook}...")
    if convert_vscode_to_jupyter(notebook):
        print(f"Successfully converted {notebook}")
    else:
        print(f"Failed to convert {notebook}")
