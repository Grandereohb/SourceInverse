import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_newman_adapter import verified_posterior


class NewmanAdapterTests(unittest.TestCase):
    def test_only_hash_verified_posterior_is_loaded_not_notebook_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notebook.ipynb'
            setup = "raise RuntimeError('must never execute setup')\n"
            notebook = dict(cells=[dict(cell_type='code', source=[setup]),
                dict(cell_type='code', source=['def log_posterior(x):\n    return scale*x\n'])])
            path.write_text(json.dumps(notebook), encoding='utf-8')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            function = verified_posterior(path, digest, {'scale': 3})
            self.assertEqual(function(2), 6)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verified_posterior(path, '0'*64, {})

    def test_ambiguous_author_posterior_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'notebook.ipynb'
            cell = dict(cell_type='code', source=['def log_posterior(x):\n    return x\n'])
            path.write_text(json.dumps(dict(cells=[cell, cell])), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                verified_posterior(path, hashlib.sha256(path.read_bytes()).hexdigest(), {})


if __name__ == '__main__':
    unittest.main()
