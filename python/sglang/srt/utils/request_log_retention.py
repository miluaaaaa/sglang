"""Time-based expiration of archives belonging to one request log instance."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)


class RequestLogRetentionHandler(TimedRotatingFileHandler):
    """Rotate hourly; expire owned archives by mtime at startup and every minute."""

    def __init__(self, filename: str, retention_days: int, *, cleanup_interval=60):
        if retention_days <= 0 or cleanup_interval <= 0:
            raise ValueError(
                "Request log retention days and cleanup interval must be positive"
            )
        super().__init__(filename, when="H", backupCount=0, encoding="utf-8")
        self.retention_seconds = retention_days * 86400
        self._cleanup_interval = cleanup_interval
        self._cleanup_stop = threading.Event()
        self.cleanup_expired_archives()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="request-log-retention", daemon=True
        )
        self._cleanup_thread.start()

    def cleanup_expired_archives(self):
        cutoff = time.time() - self.retention_seconds
        directory, basename = os.path.split(self.baseFilename)
        prefix = basename + "."
        self.acquire()
        try:
            if self._cleanup_stop.is_set():
                return
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.startswith(prefix):
                        continue
                    try:
                        datetime.strptime(entry.name[len(prefix) :], self.suffix)
                    except ValueError:
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    try:
                        if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                            os.unlink(entry.path)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning(
                            "Could not expire request log %s: %s", entry.path, exc
                        )
        except OSError as exc:
            logger.warning(
                "Could not scan request log directory %s: %s", directory, exc
            )
        finally:
            self.release()

    def _cleanup_loop(self):
        while not self._cleanup_stop.wait(self._cleanup_interval):
            self.cleanup_expired_archives()

    def doRollover(self):
        super().doRollover()
        self.cleanup_expired_archives()

    def close(self):
        self._cleanup_stop.set()
        # logging.shutdown() holds this handler's lock while calling close().
        # Joining here could deadlock with a cleanup waiting for that lock.
        super().close()
