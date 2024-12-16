import sys

from dlkit.analysis import StudyRunner
from dlkit.io.readers import read_study

config_path = (
    r"C:\Users\cluster\constantinos\mytorch\studies\bio-surrogate\config\u-cae.toml"
)
delete_old = True
args = sys.argv
if len(args) > 1:
    config_path = args[1]

config = read_study(config_path)
study = StudyRunner(config=config)

study.run()
