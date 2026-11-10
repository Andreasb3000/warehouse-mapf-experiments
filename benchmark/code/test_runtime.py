"""Short smoke recordings retain causal history and future-label coverage."""
import unittest
from runtime import evaluation_origins, host_info


class RuntimeTest(unittest.TestCase):
    def test_full_recording_keeps_reference_origins(self):
        self.assertEqual(evaluation_origins(range(0, 1800, 10), set(range(1800))),
                         [300, 700, 1100])

    def test_short_recording_uses_only_complete_windows(self):
        selected = evaluation_origins(range(0, 400, 10), set(range(400)))
        self.assertEqual(selected, [120, 160, 190])
        self.assertTrue(all(t >= 120 and t + 200 < 400 for t in selected))

    def test_insufficient_recording_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No origins'):
            evaluation_origins(range(200), set(range(200)))

    def test_host_metadata_does_not_require_macos_commands(self):
        self.assertIn('architecture', host_info())


if __name__ == '__main__':
    unittest.main()
