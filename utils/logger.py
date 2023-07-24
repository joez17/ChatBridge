import logging
import math
import utils

_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
if utils.is_main_process():
    LOGGER = logging.getLogger('__main__')  # this is the global logger
else:
    LOGGER = None