from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from loguru import logger

from . import cfg


def set_logger_tag(logger, tag):
    logger.configure(extra={"tag": tag})


logfile = f'{os.path.join(cfg["Debug"]["log_dir"], "DB")}' + '_' + '{time:YYYY-MM-DD}.log'
set_logger_tag(logger, 'DB')

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<yellow>{extra[tag]}</yellow> - <level>{message}</level>"
)
_ = logger.remove()
_ = logger.add(sys.stderr, level='DEBUG', format=logger_format)
_ = logger.add(logfile,
               level='DEBUG',
               format=logger_format,
               rotation='1 day',
               retention='90 days')