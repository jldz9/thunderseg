import sys
from pathlib import Path
sys.path.append(Path('~/thunderseg/src').expanduser().as_posix())
# -------------
# Test tool.py
# -------------
# 1. test tool.Config

from thunderseg.utils.tool import get_config
cfg = '/home/jldz9/thunderseg/test/config.toml'
a = get_config(config_path=cfg)
print(a)