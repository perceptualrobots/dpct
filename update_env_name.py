import os
import re
import json
import glob
from pathlib import Path

def update_notebook_parameters(notebook_path):
    """
    Update the DHPCTIndividual initialization parameters in a notebook.
    
    Args:
        notebook_path: Path to the notebook file
    """
    try:
        # Read the notebook content
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a backup
        backup_dir = os.path.join(os.path.dirname(notebook_path), 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(notebook_path) + '.updated2.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created backup at {backup_path}")
        
        # Check if the notebook is in VS Code Cell format
        if "<VSCode.Cell" in content:
            # Use regex to update the initialization method parameters
            # Update method signature
            pattern = r'def __init__\(\s*self\s*,\s*gym_name\s*,'
            replacement = r'def __init__(self, env_name,'
            content = re.sub(pattern, replacement, content)
            
            # Update parameter documentation
            pattern = r'- gym_name: Gym environment ID \(e\.g\. \'CartPole-v1\'\)'
            replacement = r'- env_name: String identifier for the environment (e.g. \'CartPole-v1\')'
            content = re.sub(pattern, replacement, content)
            
            # Replace self.gym_name with self.env_name
            pattern = r'self\.gym_name = gym_name'
            replacement = r'self.env_name = env_name'
            content = re.sub(pattern, replacement, content)
            
            # Update self.env = gym.make calls
            pattern = r'self\.env = gym\.make\(self\.gym_name\)'
            replacement = r'self.env = gym.make(self.env_name)'
            content = re.sub(pattern, replacement, content)
            
            # Update constructor calls
            pattern = r'DHPCTIndividual\(\s*([^,]+)\s*,'
            replacement = r'DHPCTIndividual(\1,'
            content = re.sub(pattern, replacement, content)
            
            # Update any saves/configs
            pattern = r"'gym_name': self\.gym_name,"
            replacement = r"'env_name': self.env_name,"
            content = re.sub(pattern, replacement, content)
            
            # Update config loading
            pattern = r'config\[\s*[\'"]gym_name[\'"]\s*\],'
            replacement = r'config[\'env_name\'],'
            content = re.sub(pattern, replacement, content)
            
            # Write the updated content back to the file
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated parameters in {notebook_path}")
            return True
            
        elif notebook_path.endswith('.ipynb'):
            # Handle standard Jupyter format
            try:
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Process each cell
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = ''.join(cell.get('source', []))
                        
                        # Update method signature
                        if 'def __init__(self, gym_name,' in source:
                            new_source = source.replace(
                                'def __init__(self, gym_name,', 
                                'def __init__(self, env_name,'
                            )
                            
                            # Update parameter documentation
                            new_source = re.sub(
                                r'- gym_name: .*\n',
                                r'- env_name: String identifier for the environment (e.g. \'CartPole-v1\')\n',
                                new_source
                            )
                            
                            # Replace self.gym_name with self.env_name
                            new_source = new_source.replace(
                                'self.gym_name = gym_name',
                                'self.env_name = env_name'
                            )
                            
                            # Update self.env = gym.make calls
                            new_source = new_source.replace(
                                'self.env = gym.make(self.gym_name)',
                                'self.env = gym.make(self.env_name)'
                            )
                            
                            cell['source'] = new_source.split('\n')
                        
                        # Update constructor calls
                        elif 'DHPCTIndividual(' in source and 'gym_name=' in source:
                            new_source = source.replace('gym_name=', 'env_name=')
                            cell['source'] = new_source.split('\n')
                        
                        # Update any saves/configs with gym_name
                        elif "'gym_name': self.gym_name," in source:
                            new_source = source.replace(
                                "'gym_name': self.gym_name,",
                                "'env_name': self.env_name,"
                            )
                            cell['source'] = new_source.split('\n')
                        
                        # Update config loading with gym_name
                        elif "config['gym_name']" in source:
                            new_source = source.replace(
                                "config['gym_name']",
                                "config['env_name']"
                            )
                            cell['source'] = new_source.split('\n')
                
                # Write the updated notebook back to the file
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=2)
                
                print(f"Updated parameters in {notebook_path}")
                return True
                
            except json.JSONDecodeError:
                print(f"Error: {notebook_path} is not a valid JSON file")
                return False
        
        return False
    
    except Exception as e:
        print(f"Error processing {notebook_path}: {str(e)}")
        return False

def main():
    # Define the notebook directory
    nbs_dir = "nbs"
    
    # Get all notebook files in the directory
    notebook_files = glob.glob(os.path.join(nbs_dir, "*.ipynb"))
    
    print(f"Found {len(notebook_files)} notebook files")
    print(f"Notebook files: {notebook_files}")
    
    # Update parameters in each notebook
    for notebook_path in notebook_files:
        print(f"Processing: {notebook_path}")
        try:
            # Check if file exists and has content
            file_size = os.path.getsize(notebook_path)
            print(f"  File size: {file_size} bytes")
            
            if file_size == 0:
                print(f"  Warning: {notebook_path} is empty")
                continue
                
            # Try to read the file
            with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
                first_few_bytes = f.read(100)
            print(f"  First few bytes: {repr(first_few_bytes)}")
            
            result = update_notebook_parameters(notebook_path)
            print(f"  Update result: {result}")
        except Exception as e:
            print(f"  Error processing {notebook_path}: {str(e)}")
    
    print("All notebooks processed")

if __name__ == "__main__":
    main()
