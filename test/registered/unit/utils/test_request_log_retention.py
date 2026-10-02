import argparse
import gc
import logging
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.arg_groups.arg_utils import add_cli_args_from_dataclass
from sglang.srt.arg_groups.fields.observability import Observability
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.request_log_retention import RequestLogRetentionHandler
from sglang.srt.utils.request_logger import RequestLogger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRequestLogRetention(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name) / "host_0_instance.log"

    def _archive(self, base=None, age_days=2, suffix="2000-01-01_00"):
        path = Path(str(base or self.base) + "." + suffix)
        path.write_text("request\n")
        timestamp = time.time() - age_days * 86400
        os.utime(path, (timestamp, timestamp))
        return path

    def _handler(self, **kwargs):
        handler = RequestLogRetentionHandler(str(self.base), 1, **kwargs)
        self.addCleanup(handler.close)
        return handler

    def test_startup_removes_only_expired_owned_archives(self):
        expired = self._archive()
        recent = self._archive(age_days=0, suffix="2000-01-02_00")
        other_instance = self._archive(base=self.base.with_name("host_0_other.log"))
        unrelated = self._archive(suffix="notes")
        self.base.write_text("active\n")
        handler = self._handler()
        self.assertFalse(expired.exists())
        for path in (recent, other_instance, unrelated, self.base):
            self.assertTrue(path.exists())
        self.assertEqual(handler.backupCount, 0)

    def test_idle_cleanup_runs_without_log_records(self):
        handler = self._handler(cleanup_interval=0.02)
        expired = self._archive()
        cleaned = threading.Event()
        original = handler.cleanup_expired_archives

        def cleanup():
            original()
            cleaned.set()

        with patch.object(handler, "cleanup_expired_archives", side_effect=cleanup):
            self.assertTrue(cleaned.wait(2), "idle cleanup did not run")
            self.assertFalse(expired.exists())

    def test_close_stops_cleanup_worker(self):
        handler = self._handler(cleanup_interval=60)
        handler.close()
        handler._cleanup_thread.join(timeout=2)
        self.assertFalse(handler._cleanup_thread.is_alive())
        handler.close()

    def test_close_under_handler_lock_does_not_deadlock(self):
        handler = self._handler(cleanup_interval=0.02)
        waiting = threading.Event()
        original = handler.cleanup_expired_archives

        def cleanup():
            waiting.set()
            original()

        handler.acquire()
        try:
            with patch.object(handler, "cleanup_expired_archives", side_effect=cleanup):
                self.assertTrue(waiting.wait(2))
                handler.close()
        finally:
            handler.release()
        handler._cleanup_thread.join(timeout=2)
        self.assertFalse(handler._cleanup_thread.is_alive())

    def test_request_logger_reuses_instance_and_isolates_other_instances(self):
        first = RequestLogger(True, 3, "json", [self.directory.name], 1)
        second = RequestLogger(True, 3, "json", [self.directory.name], 1)
        for request_logger in (first, second):
            self.addCleanup(request_logger.close)
        first_handler = first.targets[0].handlers[0]
        second_handler = second.targets[0].handlers[0]
        self.assertIsInstance(first_handler, RequestLogRetentionHandler)
        self.assertNotEqual(first_handler.baseFilename, second_handler.baseFilename)
        first_archive = self._archive(base=first_handler.baseFilename)
        second_archive = self._archive(base=second_handler.baseFilename)
        first_handler.cleanup_expired_archives()
        self.assertFalse(first_archive.exists())
        self.assertTrue(second_archive.exists())
        first.configure(log_requests_level=2)
        self.assertIs(first.targets[0].handlers[0], first_handler)

    def test_changing_directory_closes_removed_handler(self):
        request_logger = RequestLogger(True, 3, "json", [self.directory.name], 1)
        self.addCleanup(request_logger.close)
        old_target = request_logger.targets[0]
        old_handler = old_target.handlers[0]
        with tempfile.TemporaryDirectory() as destination:
            request_logger.configure(log_requests_target=[destination])
            old_handler._cleanup_thread.join(timeout=2)
            self.assertFalse(old_handler._cleanup_thread.is_alive())
            self.assertIsNone(old_handler.stream)
            self.assertEqual(old_target.handlers, [])
            self.assertNotEqual(request_logger.targets[0], old_target)
            request_logger.close()

    def test_discarding_logger_closes_retention_handler(self):
        request_logger = RequestLogger(True, 3, "json", [self.directory.name], 1)
        handler = request_logger.targets[0].handlers[0]
        del request_logger
        gc.collect()
        handler._cleanup_thread.join(timeout=2)
        self.assertFalse(handler._cleanup_thread.is_alive())
        self.assertIsNone(handler.stream)

    def test_default_request_logger_keeps_unlimited_retention(self):
        request_logger = RequestLogger(True, 3, "json", [self.directory.name])
        handler = request_logger.targets[0].handlers[0]
        self.addCleanup(handler.close)
        self.assertNotIsInstance(handler, RequestLogRetentionHandler)
        self.assertEqual(handler.backupCount, 0)

    def test_active_log_and_symlinks_are_preserved(self):
        self.base.write_text("active\n")
        expired_time = time.time() - 2 * 86400
        os.utime(self.base, (expired_time, expired_time))
        target = Path(self.directory.name) / "unrelated.txt"
        target.write_text("keep\n")
        link = Path(str(self.base) + ".2000-01-01_00")
        link.symlink_to(target)
        self._handler()
        self.assertTrue(self.base.exists())
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(), "keep\n")

    def test_rollover_keeps_recent_archives(self):
        handler = self._handler()
        handler.emit(logging.makeLogRecord({"msg": "request", "levelno": logging.INFO}))
        handler.doRollover()
        archives = list(self.base.parent.glob(self.base.name + ".*"))
        self.assertEqual(len(archives), 1)
        self.assertIn("request", archives[0].read_text())

    def test_retention_must_be_positive(self):
        for days in (0, -1):
            with self.subTest(days=days), self.assertRaises(ValueError):
                RequestLogRetentionHandler(str(self.base), days)

    def test_cli_retention_is_opt_in_and_rejects_nonpositive_values(self):
        parser = argparse.ArgumentParser()
        add_cli_args_from_dataclass(
            parser, Observability, fields=["log_requests_retention_days"]
        )
        self.assertIsNone(parser.parse_args([]).log_requests_retention_days)
        self.assertEqual(
            parser.parse_args(
                ["--log-requests-retention-days", "7"]
            ).log_requests_retention_days,
            7,
        )
        for days in ("0", "-1"):
            with self.subTest(days=days), self.assertRaises(SystemExit):
                parser.parse_args(["--log-requests-retention-days", days])

    def test_server_args_cli_passes_retention_to_request_logger(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        default_args = ServerArgs.from_cli_args(
            parser.parse_args(["--model-path", "unused-model"])
        )
        self.assertIsNone(default_args.log_requests_retention_days)
        configured = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "unused-model",
                    "--log-requests",
                    "--log-requests-target",
                    self.directory.name,
                    "--log-requests-retention-days",
                    "7",
                ]
            )
        )
        request_logger = RequestLogger(
            configured.log_requests,
            configured.log_requests_level,
            configured.log_requests_format,
            configured.log_requests_target,
            configured.log_requests_retention_days,
        )
        self.addCleanup(request_logger.close)
        self.assertTrue(request_logger.log_requests)
        handler = request_logger.targets[0].handlers[0]
        self.assertIsInstance(handler, RequestLogRetentionHandler)
        self.assertEqual(handler.retention_seconds, 7 * 86400)


if __name__ == "__main__":
    unittest.main()
