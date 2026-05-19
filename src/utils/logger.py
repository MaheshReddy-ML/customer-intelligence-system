import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """Create a named logger using the project logging format.

    Args:
        name: Logger name, typically the current module name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
