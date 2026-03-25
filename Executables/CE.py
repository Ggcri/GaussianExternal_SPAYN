#!/usr/bin/python3.9 -u
# Short alias wrapper for CentralExt
# Usage: CE <program> <args...>
# Example: CE mol preamble.dat ending.dat 8 16GB READ parall_n 2 R input.EIn output.EOut

import os
import sys

# Get the directory where this script is located
EXEC_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to CentralExt
CENTRALEXT_PATH = os.path.join(EXEC_DIR, "CentralExt")

# Execute CentralExt with all arguments passed through
os.execv(CENTRALEXT_PATH, ["CentralExt"] + sys.argv[1:])
