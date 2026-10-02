from __future__ import annotations

import json
import logging
import os
import socket
import sys
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional, Union

import torch.distributed as dist

from sglang.srt.utils.request_log_retention import RequestLogRetentionHandler


def create_log_targets(
    *,
    targets: Optional[List[str]],
    name_prefix: str,
    retention_days: Optional[int] = None,
    instance_id: Optional[str] = None,
) -> List[logging.Logger]:
    if retention_days is not None and retention_days <= 0:
        raise ValueError("--log-requests-retention-days must be positive")
    if not targets:
        return [_create_log_target_stdout(name_prefix)]
    if retention_days is not None and instance_id is None:
        instance_id = uuid.uuid4().hex
    return [
        _create_log_target(t, name_prefix, retention_days, instance_id) for t in targets
    ]


def _create_log_target(
    target: str,
    name_prefix: str,
    retention_days: Optional[int] = None,
    instance_id: Optional[str] = None,
) -> logging.Logger:
    if target.lower() == "stdout":
        return _create_log_target_stdout(name_prefix)
    return _create_log_target_file(target, name_prefix, retention_days, instance_id)


def _create_log_target_stdout(name_prefix: str) -> logging.Logger:
    return _create_logger_with_handler(
        f"{name_prefix}.stdout", logging.StreamHandler(sys.stdout)
    )


def _create_log_target_file(
    directory: str,
    name_prefix: str,
    retention_days: Optional[int] = None,
    instance_id: Optional[str] = None,
) -> logging.Logger:
    os.makedirs(directory, exist_ok=True)
    hostname = socket.gethostname()
    rank = dist.get_rank() if dist.is_initialized() else 0
    stem = f"{hostname}_{rank}"
    if retention_days is not None:
        stem += f"_{instance_id}"
    filename = os.path.join(directory, stem + ".log")
    if retention_days is None:
        handler = TimedRotatingFileHandler(
            filename, when="H", backupCount=0, encoding="utf-8"
        )
    else:
        handler = RequestLogRetentionHandler(filename, retention_days)
    return _create_logger_with_handler(
        f"{name_prefix}.file.{directory}.{stem}", handler
    )


def _create_logger_with_handler(name: str, handler: logging.Handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    else:
        handler.close()
    return logger


def log_json(
    loggers: Union[logging.Logger, List[logging.Logger]], event: str, data: dict
) -> None:
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data,
    }
    msg = json.dumps(log_data, ensure_ascii=False)

    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger in loggers:
        logger.info(msg)
