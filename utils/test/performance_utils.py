import logging
import time
from functools import wraps

_default_logger = logging.getLogger(__name__)

def time_it(iterations:int = 1,name:str = None,logger_instance:logging.logger = _default_logger):
    _logger_to_use = logger_instance if logger_instance is not None else _default_logger