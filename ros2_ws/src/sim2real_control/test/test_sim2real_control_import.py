"""Trivial import check for sim2real_control (no ROS runtime required)."""

import unittest


class TestPackageImport(unittest.TestCase):
    def test_package_imports(self):
        import sim2real_control

        self.assertEqual(sim2real_control.__name__, "sim2real_control")
