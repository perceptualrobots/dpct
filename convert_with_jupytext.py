import os
import glob
import subprocess

# Define the directory containing notebooks
notebook_dir = 'nbs'

# Get list of all notebook files
notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))

for notebook in notebooks:
    print(f"Converting {notebook}...")
    
    # First convert to py:percent format
    temp_py = notebook.replace('.ipynb', '.py')
    subprocess.run(['jupytext', '--to', 'py:percent', notebook, '-o', temp_py])
    
    # Then convert back to ipynb (will create proper Jupyter notebook)
    subprocess.run(['jupytext', '--to', 'notebook', temp_py, '-o', notebook])
    
    # Remove temporary py file
    os.remove(temp_py)
    print(f"Converted {notebook} successfully")

print("All notebooks converted!")
