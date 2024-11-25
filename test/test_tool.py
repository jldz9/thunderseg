
import os
os.environ['PROJ_LIB'] = '/home/jldz9/miniconda3/envs/DL/share/proj'
import sys
sys.path.append('/home/jldz9/DL/DL_packages/DLtreeseg/src')

from thunderseg.utils.tool import read_toml
from thunderseg.core.io import create_project_structure

config_path = '/home/jldz9/DL/DL_packages/DLtreeseg/src/DLtreeseg/utils/config.toml'
workdir = '/home/jldz9/DL/test'
structure = create_project_structure(workdir)
toml = read_toml(config_path)
toml.append(structure)
print()