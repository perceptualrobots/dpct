import os
import sys

# Write to a file
with open('debug_output.txt', 'w') as f:
    f.write('Python version: ' + sys.version + '\n')
    f.write('Current directory: ' + os.getcwd() + '\n')
    f.write('Script executed successfully\n')

print("Script executed successfully, check debug_output.txt")
