import sys

# Provide the path to the file
file_path = "nbs/02_individual.ipynb"

# Open the file in binary mode and read the first 100 bytes
with open(file_path, 'rb') as f:
    header = f.read(100)
    
print(f"File: {file_path}")
print(f"First 100 bytes (hex): {header.hex()}")
print(f"First 100 bytes (ascii): {repr(header)}")

# Print the text content
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
    print("\nFirst 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"{i+1}: {line.strip()}")

# Try to read as XML
print("\nAttempting to read specific XML tags:")
import re
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
    # Look for VSCode.Cell tags
    cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">'
    matches = re.findall(cell_pattern, content)
    print(f"Found {len(matches)} VSCode.Cell tags")
    for i, match in enumerate(matches[:5]):
        print(f"  {i+1}: id={match[0]}, language={match[1]}")
