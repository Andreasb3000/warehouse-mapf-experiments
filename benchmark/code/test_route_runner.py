"""Original-task ledger checks for route search, independent of plan endings."""
import json
from pathlib import Path
import tempfile
import unittest

from run_route_search import TaskHistory, geometry, build_proof_artifacts
from fleet_execution import digest


class BuildProofTests(unittest.TestCase):
    def fixture(self, directory):
        artifact = directory/'Dockerfile.parking_v4'
        artifact.write_text('FROM frozen-image\n')
        proof_path = directory/'proof.json'
        compiled = {'server/src/backup_planners/PIBT.cpp':'source-hash'}
        proof = {'image_id':'image-hash', 'compiled_source_sha256':compiled,
                 'build_artifacts_sha256':{artifact.name:digest(artifact)}}
        proof_path.write_text(json.dumps(proof))
        return artifact, proof_path, compiled

    def test_archives_the_selected_proofs_build_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, proof, compiled = self.fixture(Path(temporary))
            self.assertEqual(set(build_proof_artifacts(proof, 'image-hash', compiled)),
                             {artifact.resolve(), proof.resolve()})

    def test_mismatched_image_sources_or_changed_builder_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact, proof, compiled = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'selected image'):
                build_proof_artifacts(proof, 'wrong-image', compiled)
            with self.assertRaisesRegex(ValueError, 'source manifest'):
                build_proof_artifacts(proof, 'image-hash', {})
            artifact.write_text('FROM different-image\n')
            with self.assertRaisesRegex(ValueError, 'Build artifact differs'):
                build_proof_artifacts(proof, 'image-hash', compiled)


class RouteLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'native.log'
        self.tasks = [{'robot_id': 0, 'release_at_s': 0., 'stages': [
            {'location': [1, 2], 'operation': 'pickup'},
            {'location': [4, 5], 'operation': 'dropoff'}]}]
        self.history = TaskHistory(self.path, self.tasks)

    def append(self, event, newline=True):
        with self.path.open('a') as stream:
            stream.write(json.dumps(event) + ('\n' if newline else ''))

    def goal(self, row, col, clock=10):
        return dict(type='goal_reached', agent=0, row=row, col=col, clock=clock)

    def test_only_actual_final_stage_marks_robot_complete(self):
        self.append({'type': 'plan_snapshot', 'robots': [{'id': 0, 'actions': []}]})
        self.history.update()
        self.assertEqual(self.history.completed_robot_ids, set())
        self.append(self.goal(1, 2)); self.history.update()
        self.assertEqual(self.history.completed_robot_ids, set())
        self.append(self.goal(4, 5, 20)); self.history.update()
        self.assertEqual(self.history.completed_robot_ids, {0})
        self.assertEqual(self.history.goals, 2)
        self.assertEqual(self.history.last_ack_clock, 20)
        self.history.update()
        self.assertEqual(self.history.goals, 2)

    def test_partial_log_line_is_not_read_early(self):
        self.append(self.goal(1, 2), newline=False)
        self.history.update(); self.assertEqual(self.history.goals, 0)
        with self.path.open('a') as stream:
            stream.write('\n')
        self.history.update(); self.assertEqual(self.history.goals, 1)

    def test_wrong_stage_and_duplicate_are_rejected(self):
        self.append(self.goal(4, 5))
        with self.assertRaises(ValueError):
            self.history.update()
        self.path.unlink(); self.history = TaskHistory(self.path, self.tasks)
        self.append(self.goal(1, 2)); self.append(self.goal(4, 5, 20)); self.append(self.goal(4, 5, 21))
        with self.assertRaises(ValueError):
            self.history.update()

    def test_route_geometry_ignores_only_timing(self):
        a = {'plan': {'plan': [[[1, 2, 0., -1], [3, 4, 1., 7]]]}}
        b = {'plan': {'plan': [[[1, 2, 2., -1], [3, 4, 8., 7]]]}}
        self.assertEqual(geometry(a), geometry(b))
        b['plan']['plan'][0].insert(1, [1, 2, 3., -1])
        self.assertEqual(geometry(a), geometry(b))
        b['plan']['plan'][0][-1][-1] = 8
        self.assertNotEqual(geometry(a), geometry(b))


if __name__ == '__main__':
    unittest.main()
