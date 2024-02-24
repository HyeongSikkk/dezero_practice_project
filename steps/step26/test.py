import unittest
import core
import numpy as np

# do test -> python -m unittest test.py

class SquareTest(unittest.TestCase) :
    # 순전파 테스트
    def test_forward(self) :
        x = core.Variable(np.array(2.0))
        y = core.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    # 역전파 테스트
    def test_backward(self) :
        x = core.Variable(np.array(3.0))
        y = core.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self) :
        x = core.Variable(np.random.rand(1))
        y = core.square(x)
        y.backward()
        num_grad = core.numerical_diff(core.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)