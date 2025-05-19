import os
import sys
import json
import subprocess
from pathlib import Path

def fix_notebook_for_nbdev(notebook_path):
    """Fix a notebook to make it compatible with nbdev_export."""
    print(f"Fixing {notebook_path} for nbdev_export...")
    
    # Read notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse as JSON
        notebook = json.loads(content)
        
        # Ensure all code cells have execution_count and outputs
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['execution_count'] = None
                if 'outputs' not in cell:
                    cell['outputs'] = []
        
        # Write back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
            
        print(f"Fixed {notebook_path}")
        return True
    except Exception as e:
        print(f"Error fixing {notebook_path}: {str(e)}")
        return False

def export_notebook(notebook_path):
    """Try to export a single notebook using nbdev."""
    print(f"Exporting {notebook_path}...")
    
    # First, fix the notebook
    if not fix_notebook_for_nbdev(notebook_path):
        return False
    
    # Try to export
    try:
        cmd = ["nbdev_export", "--path", notebook_path]
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Standard output: {result.stdout}")
        if result.stderr:
            print(f"Standard error: {result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"Error exporting {notebook_path}: {str(e)}")
        return False

def main():
    # Check if an argument was provided
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # Export all notebooks
        notebooks_dir = os.path.abspath("nbs")
        notebook_files = [os.path.join(notebooks_dir, f) for f in os.listdir(notebooks_dir) 
                         if f.endswith('.ipynb') and not f.startswith('fixed_') and not f.startswith('jupyter_')]
        
        successful = 0
        failed = 0
        
        for notebook_path in notebook_files:
            if export_notebook(notebook_path):
                print(f"Successfully exported {notebook_path}")
                successful += 1
            else:
                print(f"Failed to export {notebook_path}")
                failed += 1
        
        print(f"\nExport summary: {successful} successful, {failed} failed")
    else:
        # Export just one notebook
        notebook_path = os.path.abspath("nbs/00_core.ipynb")
        
        # Export
        if export_notebook(notebook_path):
            print(f"Successfully exported {notebook_path}")
        else:
            print(f"Failed to export {notebook_path}")

if __name__ == "__main__":
    main()
