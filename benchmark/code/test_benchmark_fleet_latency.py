"""Fresh timing protocols use model identities, not inferred filesystem paths."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_fleet_latency import make_runtime_protocol, validate_model_protocol


class RuntimeProtocolTests(unittest.TestCase):
    def test_fresh_protocol_preserves_named_hashes_and_available_origins(self):
        hashes = {'local_ridge': 'ridge', 'convlstm': 'neural', 'residual_convlstm': 'residual'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction = root/'extraction.json'
            extraction.write_text(json.dumps({'protocol': {'first_origin_s': 12, 'last_origin_s': 16,
                                                          'origin_stride_s': 4},
                                             'identity': {'model_sha256': hashes}}))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'runs': [{'workload_seed': 8301}]}))
            protocol = make_runtime_protocol(extraction, [manifest])
            self.assertEqual(protocol['runtime']['origin_s'], [12, 16])
            self.assertEqual(protocol['model_hash_key'], 'name')
            self.assertEqual(protocol['model_sha256'], hashes)
            with patch('benchmark_fleet_latency.model_hashes', return_value=hashes):
                validate_model_protocol(protocol)
                protocol['model_sha256'] = {**hashes, 'local_ridge': 'changed'}
                with self.assertRaisesRegex(ValueError, 'Frozen model changed'):
                    validate_model_protocol(protocol)

    def test_missing_model_is_not_accepted(self):
        with patch('benchmark_fleet_latency.model_hashes', return_value={'local_ridge': 'correct'}):
            with self.assertRaisesRegex(ValueError, 'Frozen model changed'):
                validate_model_protocol({'model_hash_key': 'name', 'model_sha256': {}})


if __name__ == '__main__':
    unittest.main()
