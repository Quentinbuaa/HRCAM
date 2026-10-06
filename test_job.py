"""CPU binding policy and real subprocess inheritance on remote Linux."""
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from job import SAFE_CPUS, bind_safe_cpus


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity required")
class AffinityTests(unittest.TestCase):
    def test_selects_only_approved_cpus(self):
        with patch("job.os.sched_getaffinity", side_effect=[set(range(32)), set(SAFE_CPUS)]), patch("job.os.sched_setaffinity") as setter:
            self.assertEqual(bind_safe_cpus(), list(range(24, 32)))
            setter.assert_called_once_with(0, set(SAFE_CPUS))

    def test_respects_existing_restriction(self):
        with patch("job.os.sched_getaffinity", return_value={25, 26}), patch("job.os.sched_setaffinity") as setter:
            self.assertEqual(bind_safe_cpus(), [25, 26])
            setter.assert_called_once_with(0, {25, 26})

    def test_fails_closed_without_safe_cpus(self):
        with patch("job.os.sched_getaffinity", return_value={12, 13}), patch("job.os.sched_setaffinity") as setter:
            with self.assertRaises(RuntimeError):
                bind_safe_cpus()
            setter.assert_not_called()

    def test_real_child_inherits_binding(self):
        original = os.sched_getaffinity(0)
        if not original & SAFE_CPUS:
            self.skipTest("Remote-5080 approved CPUs unavailable")
        try:
            selected = bind_safe_cpus()
            output = subprocess.check_output([
                sys.executable, "-c",
                "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))",
            ], text=True)
            self.assertEqual(json.loads(output), selected)
        finally:
            os.sched_setaffinity(0, original)


if __name__ == "__main__":
    unittest.main()
