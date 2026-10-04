import os

# Check if ROOT_DIR is set as an environment variable; otherwise, use the default
#ROOT_DIR = os.environ.get('CUSTOM_ROOT_DIR', os.path.dirname(os.path.abspath(__file__)))
CUSTOM_ROOT_DIR = os.environ.get("CUSTOM_ROOT_DIR", os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set the DATA_DIR to the directory where your data resides.
DATA_DIR = os.path.join(ROOT_DIR, 'data')