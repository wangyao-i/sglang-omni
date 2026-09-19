"""CPU checks of coordination; no torch import and no device qualification."""
import importlib.util
from pathlib import Path
import threading
import unittest
import subprocess
import sys
import shutil
import tempfile
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "cycle_probe", Path(__file__).with_name("probe_graph_h2d_cycle.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class CoordinationTests(unittest.TestCase):
    def test_loader_preserves_path_and_main_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            helper = Path(temp) / "helper with 'quote'.py"
            helper.write_text("assert __name__ == '__main__'\nprint('LOADER_OK')\n", encoding="utf-8")
            command = probe.snapshot_command(123, helper)
            self.assertEqual(command.count("detach"), 0)  # Helper owns detach.
            code = command[-1].removeprefix("python ")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("LOADER_OK", result.stdout)

    @unittest.skipUnless(shutil.which("gdb"), "real GDB unavailable locally")
    def test_real_gdb_executes_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            helper = Path(temp) / "helper with 'quote'.py"
            helper.write_text("assert __name__ == '__main__'\nprint('GDB_LOADER_OK')\n", encoding="utf-8")
            load = probe.snapshot_command(123, helper)[-1]
            result = subprocess.run(["gdb", "-nx", "-nh", "-batch", "-ex", load],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GDB_LOADER_OK", result.stdout)

    def test_update_can_release_blocked_copy(self):
        released = threading.Event()
        phases = []

        def copy():
            if not released.wait(1):
                raise AssertionError("update incorrectly waited for copy")

        probe.overlap(copy, released.set, 0, log=phases.append)
        self.assertEqual(phases, ["copy.enter", "copy.return"])

    def test_copy_failure_is_not_success(self):
        def copy():
            raise ValueError("copy failed")

        with self.assertRaisesRegex(ValueError, "copy failed"):
            probe.overlap(copy, lambda: None, 0, log=lambda _: None)

    def test_workers_are_distinct(self):
        ids = []
        probe.overlap(lambda: ids.append(threading.get_native_id()),
                      lambda: ids.append(threading.get_native_id()),
                      0, log=lambda _: None)
        self.assertEqual(len(set(ids)), 2)

    def test_parent_times_out_and_reaps_real_cpu_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                child.wait(timeout=0.1)
            self.assertEqual(probe.stop_owned(child), "terminated")
            self.assertIsNotNone(child.poll())
        finally:
            probe.stop_owned(child)

    def test_cleanup_runs_kill_after_terminate_timeout(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("owned", 5), 0]
        self.assertEqual(probe.stop_owned(child), "killed")
        child.terminate.assert_called_once()
        child.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
