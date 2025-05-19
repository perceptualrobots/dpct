import os
import json
import glob
import re
import nbformat

def convert_vscode_to_jupyter(file_path):
    """
    Convert a VS Code notebook to Jupyter format with parameter name standardization.
    
    Args:
        file_path: Path to the VS Code notebook file
        
    Returns:
        Path to the converted notebook file or None if conversion failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if this is a VS Code notebook
        if "<VSCode.Cell" not in content:
            print(f"{file_path} is not a VS Code notebook")
            return None
        
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
                # Replace gym_name with env_name in code cells
                cell_content = re.sub(r'self\.gym_name', 'self.env_name', cell_content)
                cell_content = re.sub(r'env_name,\s*gym_name', 'env_name', cell_content)
                cell_content = re.sub(r'"env_name":\s*self\.env_name,\s*"gym_name":\s*self\.gym_name', 
                                   '"env_name": self.env_name', cell_content)
                cell_content = re.sub(r'config\[\'env_name\'\],\s*config\[\'gym_name\'\]', 
                                   'config[\'env_name\']', cell_content)
                cell_content = re.sub(r'offspring\s*=\s*DHPCTIndividual\(\s*self\.env_name,\s*self\.gym_name', 
                                   'offspring = DHPCTIndividual(self.env_name', cell_content)
                                   
                cell = nbformat.v4.new_code_cell(cell_content)
            
            # Add metadata
            if props['id']:
                if 'metadata' not in cell:
                    cell['metadata'] = {}
                cell['metadata']['id'] = props['id']
                
            nb.cells.append(cell)
            
        # Write the notebook
        output_path = file_path.replace('.ipynb', '_jupyter.ipynb')
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"Converted {file_path} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

def main():
    # Find all notebook files
    notebooks_dir = "C:/Users/ruper/Versioning/python/nbdev/dpct/nbs"
    print(f"Looking for notebooks in {notebooks_dir}")
    notebook_files = glob.glob(os.path.join(notebooks_dir, "*.ipynb"))
    print(f"Found {len(notebook_files)} notebook files")
    
    # Process each notebook
    for notebook_file in notebook_files:
        # Skip already converted notebooks
        if "_jupyter" in notebook_file or "fixed_" in notebook_file:
            print(f"Skipping {notebook_file} (already converted)")
            continue
        
        print(f"Processing {notebook_file}...")
        jupyter_notebook = convert_vscode_to_jupyter(notebook_file)
        
        if jupyter_notebook:
            # Backup the original file
            backup_dir = os.path.join(notebooks_dir, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(backup_dir, os.path.basename(notebook_file))
            try:
                with open(notebook_file, 'r', encoding='utf-8') as f_in:
                    with open(backup_file, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
                
                # Replace the original file with the Jupyter notebook
                with open(jupyter_notebook, 'r', encoding='utf-8') as f_in:
                    with open(notebook_file, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
                
                print(f"Replaced {notebook_file} with Jupyter format")
                
                # Clean up the temporary file
                os.remove(jupyter_notebook)
            except Exception as e:
                print(f"Error replacing {notebook_file}: {e}")

if __name__ == "__main__":
    main()
