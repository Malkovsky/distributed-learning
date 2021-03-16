import logging


def get_logger(log_file_path='logs.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_hms(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
