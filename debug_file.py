import os

# Define the notebook to debug
notebook_path = 'nbs/02_individual.ipynb'

# Check file existence
if not os.path.exists(notebook_path):
    print(f"ERROR: File {notebook_path} does not exist")
else:
    print(f"File {notebook_path} exists")

# Check file size
file_size = os.path.getsize(notebook_path)
print(f"File size: {file_size} bytes")

# Try reading the file byte by byte
try:
    with open(notebook_path, 'rb') as f:
        header = f.read(100)
        print(f"First 100 bytes (hex): {header.hex()}")
        print(f"First 100 bytes (ascii): {repr(header)}")
except Exception as e:
    print(f"Error reading file: {str(e)}")

# Try opening the file in text mode
try:
    with open(notebook_path, 'r', encoding='utf-8', errors='replace') as f:
        first_few_lines = []
        for i, line in enumerate(f):
            if i < 10:
                first_few_lines.append(line.strip())
        print("First 10 lines:")
        for i, line in enumerate(first_few_lines):
            print(f"{i+1}: {line}")
except Exception as e:
    print(f"Error reading file in text mode: {str(e)}")
