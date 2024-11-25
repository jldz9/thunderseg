import sys
import time
sys.path.append('/home/vscode/remotehome/DL_packages/thunderseg/src')

from thunderseg import main
main.main(['--init', '/home/vscode/remotehome/DL_drake/output'])
main.main(['preprocess', '-c', '/home/vscode/remotehome/DL_drake/output'])
#time.sleep(5)
#main.main(['train', '-c', '/home/vscode/remotehome/DL_drake/output'])
#TODO Bug? When run sequently has Unexpected segmentation fault encountered in worker error, but if run only trian does not 
main.main(['predict', '-c', '/home/vscode/remotehome/DL_drake/output'])