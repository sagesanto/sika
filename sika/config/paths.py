import os
from os.path import abspath, dirname, join, exists, pardir

libdir = dirname(abspath(__file__))

config_path = join(libdir, "config.toml")
logging_config_path = join(libdir, "logging.json")
logging_config_path_nofile = join(libdir, "logging_nofile.json")

logging_dir = abspath(join(libdir,pardir,"logs"))
os.makedirs(logging_dir,exist_ok=True)