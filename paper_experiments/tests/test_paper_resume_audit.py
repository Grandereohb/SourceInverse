import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_paper_solver_upgrade import validate_resume


class ResumeAuditTests(unittest.TestCase):
    def test_audit_only_permits_declared_source_changes(self):
        payload = b'frozen protocol'
        saved = dict(methods=['pso'], budget=256, inputs='unchanged',
                     source_sha256={'driver': 'old', 'forward': 'same'})
        current = dict(saved, source_sha256={'driver': 'new', 'forward': 'same'})
        audit = dict(frozen_protocol_sha256=hashlib.sha256(payload).hexdigest(),
                     approved_source_changes={'driver': dict(before='old', after='new')})
        validate_resume(saved, current, audit, payload)
        for key, value in [('budget', 64), ('inputs', 'changed'), ('methods', ['ia'])]:
            changed = dict(current, **{key: value})
            with self.assertRaises(ValueError):
                validate_resume(saved, changed, audit, payload)
        changed = copy.deepcopy(current)
        changed['source_sha256']['forward'] = 'changed'
        with self.assertRaises(ValueError):
            validate_resume(saved, changed, audit, payload)
        with self.assertRaises(ValueError):
            validate_resume(saved, current, audit, b'other frozen protocol')


if __name__ == '__main__':
    unittest.main()
