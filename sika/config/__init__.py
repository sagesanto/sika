import os, shutil, glob
from os.path import abspath, dirname, join, exists, pardir
import tomlkit
dir = dirname(abspath(__file__))

from sika.config.config import Config
from sika.config.utils import configure_logger

from sika.config.paths import config_path, logging_dir


# adding more directories here will allow .default replacement in those directories as well
config_dirs = [dir,]

# recursively search for dirs containing ".default" files
config_dirs = set([dirname(abspath(f)) for f in glob.glob(join(dir,pardir,"**","*.default"),recursive=True)])
# print("Found .default files in the following dirs:", ", ".join(config_dirs))
for d in config_dirs:
    for f in [f for f in os.listdir(d) if ".default" in f]:
        active = join(d,f.replace(".default",""))
        if not exists(active):
            shutil.copy(join(d,f),active)
        elif active.endswith("toml"):
            with open(join(d,f),"rb") as default:
                default_cfg = tomlkit.load(default)
            with open(active,"rb") as ac:
                active_cfg = tomlkit.load(ac)
            def update_cfg(default, active,path="",paths=[]):
                for k, v in default.items():
                    if k not in active:
                        active[k] = v
                        paths.append(path + k)
                    elif isinstance(v, dict):
                        paths = update_cfg(v, active[k],path+k+".",paths)
                return paths
            key_paths = update_cfg(default_cfg,active_cfg)
            if key_paths:
                with open(active,"w") as f:
                    f.write(tomlkit.dumps(active_cfg).replace("\r\n","\n"))
                logger = configure_logger("config")
                logger.warn(f"The default config '{join(d,f)}' has new keys that were not present in the active config '{active}'. Default values were copied over for the following new keys: {key_paths}")

# with open(config_path,"rb") as f:
#     config = tomlkit.load(f)