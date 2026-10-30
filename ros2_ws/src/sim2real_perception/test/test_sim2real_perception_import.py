"""Trivial import check for sim2real_perception (no ROS runtime required)."""

import unittest


class TestPackageImport(unittest.TestCase):
    def test_package_imports(self):
        import sim2real_perception

        self.assertEqual(sim2real_perception.__name__, "sim2real_perception")
