import logging
import os 

def setup_logger(name = __name__, log_file = 'MRI_bib.log', level = logging.INFO):
    """
    Set up a logger with the given name and level.

    Parameters
    ----------
    name : str, optional
        Name of the logger (default is the module name).
    log_file : str, optional
        File path for the log file (default is 'MRI_bib.log').
    level : int, optional
        Logging level (default is logging.INFO).

    Returns
    -------
    logger : logging.Logger
        Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure directory exists 

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File Handler 
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # making the logger 
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoiding multiple handlers 
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    return logger


