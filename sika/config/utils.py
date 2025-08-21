import os
import json
import logging
from pathlib import Path
import logging.config
from os.path import join, dirname, abspath, pardir
import tomlkit
from typing import List, Any

from sika.config.paths import logging_config_path, logging_dir

def configure_logger(name, log_dir=None):
    """Get a logger.

    :param name: name of this logger. the log file will be saved at ``log_dir``/``name``.log
    :type name: str
    :param log_dir: the directory to save logs to. None (default) will use the sika logging directory (``sika.config.paths.logging_dir``).
    :type outfile_path: str, optional
    :return: logger
    :rtype: logging.Logger
    """
    
    if log_dir == None:
        log_dir = logging_dir
    # first, check if the logger has already been configured
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)
    try:
        with open(logging_config_path, 'r') as log_cfg:
            logging.config.dictConfig(json.load(log_cfg))
            logger = logging.getLogger(name)
            # set outfile of existing filehandler. need to do this instead of making a new handler in order to not wipe the formatter off
            # NOTE RELIES ON FILE HANDLER BEING THE SECOND HANDLER
            root_logger = logging.getLogger()
            outfile_path = join(log_dir, name+".log")
            file_handler = root_logger.handlers[1]
            file_handler.setStream(Path(outfile_path).open('a'))
            try:
                os.remove("should_be_set_by_code.log")  # pardon this
            except:
                pass

    except Exception as e:
        print(f"Can't load logging config ({e}). Using default config.")
        logger = logging.getLogger(name)
        file_handler = logging.FileHandler(outfile_path, mode="a+")
        logger.addHandler(file_handler)

    # install_mp_handler()
    return logger