"""Native server scheduling labels must not shift the physical pose history."""
import json
from pathlib import Path
import tempfile
import unittest

from run_decisions import LiveHistory


def frame(clock, row=None):
    return {"type": "tick", "clock": clock,
            "agents": [{"id": 0, "x": -(clock if row is None else row), "y": -3, "rz": 0}]}


def request():
    return {"clock": 120, "observation_clock": 121, "pose_frame": frame(121),
            "snapshot": {"clock": 121, "server_clock": 120},
            "controllers": [{"clock": 121, "poll_phase": 0}]}


class LiveHistoryPhaseTest(unittest.TestCase):
    def test_physical_frame_120_is_included_and_future_frames_never_enter_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulator.log"
            events = [frame(clock, 999 if clock >= 121 else None) for clock in range(131)]
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            past = LiveHistory(path).past(request())
            self.assertEqual(past.shape, (121, 1, 3))
            self.assertEqual(past[0, 0, 0], 1)
            self.assertEqual(past[-2, 0, 0], 120)
            self.assertEqual(past[-1, 0, 0], 121)
            self.assertFalse((past[:, 0, 0] == 999).any())

    def test_missing_latest_causal_physics_frame_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulator.log"
            path.write_text("".join(json.dumps(frame(clock)) + "\n" for clock in range(120)))
            with self.assertRaises(ValueError):
                LiveHistory(path).past(request())


if __name__ == "__main__":
    unittest.main()
