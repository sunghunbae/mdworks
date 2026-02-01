import sys
import logging
from pathlib import Path


def setup_logger(logger: logging.Logger, 
                 workdir: Path, 
                 prefix: Path, 
                 quiet: bool = False) -> None:
    
    logger.setLevel(logging.DEBUG) # first filter

    logger_format = logging.Formatter(
        fmt='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M',
        )
    
    # console handler
    stdout_stream_handler = logging.StreamHandler(sys.stdout)
    stdout_stream_handler.setLevel(logging.DEBUG)
    stdout_stream_handler.setFormatter(logger_format)

    # file handler
    logging_file_handler = logging.FileHandler(workdir / f"{prefix}.log")
    logging_file_handler.setLevel(logging.DEBUG)
    logging_file_handler.setFormatter(logger_format)
    
    # check existing handlers
    is_filehandler_attached = False
    is_streamhandler_attached = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            is_filehandler_attached = True
        if isinstance(handler, logging.StreamHandler):
            is_streamhandler_attached = True

    # attach FileHandler if necessary
    if not is_filehandler_attached:
        logger.addHandler(logging_file_handler)

    if quiet:
        # FileHandler only
        if is_streamhandler_attached:
            logger.removeHandler(stdout_stream_handler)
    else:
        # FileHandler + StreamHandler
        if not is_streamhandler_attached:
            logger.addHandler(stdout_stream_handler)