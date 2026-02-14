import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a named logger that outputs logs to the console (stdout).

    This helper centralizes logging configuration so all modules in the project can
    use a consistent log format and log level. It uses `logging.basicConfig()` to
    configure the root logger with a StreamHandler pointing to standard output.

    Key behaviors:
    - Logs are printed to stdout (useful for Docker/Kubernetes logs and CI).
    - The log format includes timestamp, level, and message.
    - `force=True` resets existing handlers to prevent duplicate logs when this
      function is called multiple times (common in notebooks/tests).

    Args:
        name (str):
            The logger name. Typically use `__name__` for module-level loggers,
            or a file/module identifier for easy tracing in multi-module projects.
        level (int):
            The logging severity level. Defaults to `logging.INFO`.
            Common values: logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL.

    Returns:
        logging.Logger:
            A configured logger instance with the specified name and level.
    """

    # Configure the root logger:
    # - level=level: show messages at `level` and above (e.g., INFO+).
    # - format=...: include timestamp, severity, and the log message.
    # - handlers=[StreamHandler(sys.stdout)]: send logs to stdout explicitly.
    # - force=True: remove any existing handlers and reconfigure from scratch.
    #   This prevents duplicated log lines if basicConfig was already called elsewhere.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Create (or retrieve) a named logger. Using named loggers makes it easier
    # to trace which module produced the log.
    logger = logging.getLogger(name)

    # Ensure the logger emits at the desired level (in addition to root config).
    logger.setLevel(level)

    return logger
