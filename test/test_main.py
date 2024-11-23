import sys
sys.path.append('/home/vscode/remotehome/DL_packages/DLtreeseg/src')

from DLtreeseg import main

main.main(['preprocess', '-c', '/workspaces/DLtreeseg/src/DLtreeseg/utils/config.toml'])