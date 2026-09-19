"""CPU-only policy tests; not an attach or unwinding qualification."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import queue_snapshot_gdb as collector
from queue_snapshot_gdb import LIB_SHA, ordered_threads, parse_tids, repository_state


class SnapshotPolicyTests(unittest.TestCase):
    def test_full_maps_flushed_even_when_library_inspection_fails(self):
        class Inferior:
            pid = 123
        class Gdb:
            def selected_inferior(self): return Inferior()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"
            proc.mkdir()
            maps = "1000-2000 r-xp 0 00:00 0 /runtime/implementation.so\n"
            (proc / "maps").write_text(maps)
            output = root / "snapshot.jsonl"
            def route(value):
                return proc if str(value) == "/proc/123" else Path(value)
            with patch.dict("os.environ", {"WF_QUEUE_TIDS": "1", "WF_QUEUE_SECONDS": "15",
                                           "WF_QUEUE_SNAPSHOT": str(output)}), \
                 patch.object(collector, "Path", side_effect=route), \
                 patch.object(collector, "matching_library", side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    collector.snapshot(Gdb())
            row = json.loads(output.read_text())
            self.assertEqual(row, {"phase": "maps", "pid": 123, "text": maps})

    def test_priority_not_descending_gdb_number(self):
        rows = [{"tid": 9, "name": "python"}, {"tid": 2, "name": "acl_thread"},
                {"tid": 3, "name": "release_thread"}, {"tid": 4, "name": "acl_thread"},
                {"tid": 1, "name": "python"}]
        ordered, _ = ordered_threads(rows, {9})
        self.assertEqual([row["tid"] for row in ordered], [9, 2, 3, 4, 1])

    def test_tids_required_positive(self):
        for value in ("", "0", "-1", "old-tid"):
            with self.assertRaises(ValueError):
                parse_tids(value)
        self.assertEqual(parse_tids("9,10,9"), {9, 10})

    def test_wrong_library_never_reads_frame(self):
        self.assertIsNone(repository_state(None, None, {"sha256": "wrong"}))
        self.assertIsNone(repository_state(None, None, None))

    def test_matching_frame_decodes_only_selected_fields(self):
        class Arch:
            def name(self): return "aarch64"
        class Frame:
            def architecture(self): return Arch()
            def pc(self): return 0x1000000 + 0xCA54B0
            def read_register(self, name):
                self_name = name
                assert self_name == "x19"
                return 0x2000
        class Inferior:
            def read_memory(self, address, size):
                assert 0x2000 <= address < 0x2060
                return (1).to_bytes(size, "little")
        state = repository_state(Frame(), Inferior(), {"sha256": LIB_SHA, "base": 0x1000000})
        self.assertEqual(state["mutex_address"], "0x2060")
        self.assertEqual(state["need_empty"], 1)
        self.assertIn("unknown", state["mutex_owner"])


if __name__ == "__main__":
    unittest.main()
