import os
import json
from pathlib import Path

# Path to a notebook we want to examine
notebook_path = "nbs/02_individual.ipynb"

# Read the notebook
try:
    print(f"Examining {notebook_path}...")
    
    # Check file properties
    if os.path.exists(notebook_path):
        print(f"File exists: Yes")
        file_size = os.path.getsize(notebook_path)
        print(f"File size: {file_size} bytes")
    else:
        print(f"File exists: No")
        exit(1)
    
    # Try to read the file
    with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        
    print(f"Content length: {len(content)} characters")
    print(f"First 100 characters: {repr(content[:100])}")
    
    # Parse as JSON
    try:
        notebook = json.loads(content)
        print(f"Successfully parsed as JSON")
        print(f"Has cells: {'cells' in notebook}")
        if 'cells' in notebook:
            print(f"Number of cells: {len(notebook['cells'])}")
            
            # Look for `gym_name` occurrences
            cells_with_gym_name = 0
            for i, cell in enumerate(notebook['cells']):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    if 'gym_name' in source:
                        cells_with_gym_name += 1
                        print(f"\nFound 'gym_name' in cell {i}:")
                        print(f"Cell source (excerpt): {source[:150]}...")
            
            print(f"\nTotal cells with 'gym_name': {cells_with_gym_name}")
            
            # Modify the notebook
            print("\nUpdating notebook...")
            
            # Create a backup
            backup_dir = os.path.join(os.path.dirname(notebook_path), 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(notebook_path) + '.direct_edit.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Created backup at {backup_path}")
            
            # Process each cell
            cells_updated = 0
            for cell in notebook['cells']:
                if cell.get('cell_type') == 'code':
                    source = [line for line in cell.get('source', [])]
                    new_source = []
                    
                    for line in source:
                        # Replace parameter signature
                        if 'def __init__(self,' in line and 'gym_name,' in line:
                            new_line = line.replace('gym_name,', 'env_name,')
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated init signature: {line} -> {new_line}")
                        
                        # Replace parameter documentation
                        elif '- gym_name:' in line:
                            new_line = line.replace('- gym_name:', '- env_name:')
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated parameter doc: {line} -> {new_line}")
                        
                        # Replace self.gym_name assignment
                        elif 'self.gym_name = gym_name' in line:
                            new_line = line.replace('self.gym_name = gym_name', 'self.env_name = env_name')
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated attribute assignment: {line} -> {new_line}")
                        
                        # Replace gym.make call
                        elif 'self.env = gym.make(self.gym_name)' in line:
                            new_line = line.replace('self.env = gym.make(self.gym_name)', 'self.env = gym.make(self.env_name)')
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated gym.make call: {line} -> {new_line}")
                        
                        # Replace constructor calls
                        elif 'DHPCTIndividual(' in line and 'gym_name=' in line:
                            new_line = line.replace('gym_name=', 'env_name=')
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated constructor call: {line} -> {new_line}")
                        
                        # Replace configuration dictionary
                        elif "'gym_name':" in line:
                            new_line = line.replace("'gym_name':", "'env_name':")
                            new_source.append(new_line)
                            cells_updated += 1
                            print(f"Updated config dict: {line} -> {new_line}")
                        
                        else:
                            new_source.append(line)
                    
                    cell['source'] = new_source
            
            print(f"\nTotal lines updated: {cells_updated}")
            
            # Write the updated notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)
            
            print(f"Successfully updated and saved {notebook_path}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
    
except Exception as e:
    print(f"Error: {str(e)}")
