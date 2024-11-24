import sys
import time
sys.path.append('/home/vscode/remotehome/DL_packages/DLtreeseg/src')

from DLtreeseg import main
#main.main(['--init', '/workspaces/DLtreeseg/test'])
#main.main(['preprocess', '-c', '/workspaces/DLtreeseg/test'])
#time.sleep(5)
main.main(['train', '-c', '/workspaces/DLtreeseg/test/config.toml'])
#TODO Bug? When run sequently has Unexpected segmentation fault encountered in worker error, but if run only trian does not 