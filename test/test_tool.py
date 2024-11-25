
import os
os.environ['PROJ_LIB'] = '/home/jldz9/miniconda3/envs/DL/share/proj'
import sys
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

from thunderseg.utils.tool import read_toml

config_path = '/home/jldz9/DL/DL_packages/thunderseg/src/thunderseg/utils/config.toml'
workdir = '/home/jldz9/DL/test'
structure = create_project_structure(workdir)
toml = read_toml(config_path)
toml.append(structure)
print()