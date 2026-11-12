import unittest
import numpy as np
from ia_3d_affine_transport import general_step
from ia_3d_affine_fast import shared_gram_step


class SharedGramTests(unittest.TestCase):
    def test_shared_gram_against_reference(self):
        matrices = [np.zeros((2, 2)), np.array([[.01, .02], [-.01, .005]]),
                    np.array([[0., .03], [0., 0.]]), np.array([[.03, 0.], [0., -.03]]),
                    np.array([[0., .03], [-.03, 0.]])]
        for a in matrices:
            for order in [16, 32]:
                arguments = (a, np.array([2., -1.]), np.array([0., .001, 1., 20., 60.]), np.array([.6, .8]))
                reference = general_step(*arguments, order=order)
                actual = shared_gram_step(*arguments, order=order)
                for k in [0, 1, 4, 5]:
                    np.testing.assert_allclose(actual[k], reference[k], rtol=1e-13, atol=1e-13)
                for k in [2, 3]:
                    np.testing.assert_allclose(actual[k] @ actual[k].swapaxes(-1, -2),
                        reference[k] @ reference[k].swapaxes(-1, -2), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
