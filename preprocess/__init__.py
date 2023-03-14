import sys
import os

modules = [
    os.path.abspath('../clip_server')
]

for module in modules:
    if module not in sys.path:
        sys.path.append(module)

