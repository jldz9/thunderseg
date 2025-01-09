import sys
import time
sys.path.append('/home/jldz9/DL/DL_packages/thunderseg/src')

from thunderseg import main
main.main(['--init', '/home/jldz9/DL/DL_drake/output'])
main.main(['preprocess', '-c', '/home/jldz9/DL/config.toml'])
#time.sleep(5)
main.main(['train', '-c', '/home/jldz9/DL/config.toml'])
#TODO Bug? When run sequently has Unexpected segmentation fault encountered in worker error, but if run only trian does not 
main.main(['predict', '-c', '/home/jldz9/DL/config.toml'])