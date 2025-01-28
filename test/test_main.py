import sys
import time
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

from thunderseg import main
from types import SimpleNamespace

args = SimpleNamespace()
args.config = '/home/jldz9/thunderseg_example_data/config.toml'

#main.preprocess_step(args)
#main.train_step(args)
main.predict_step(args)