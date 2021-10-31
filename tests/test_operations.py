import unittest

from app.tensor import *

class TestTensorSum (unittest.TestCase):
    def test_sum_without_grad (self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = t1.sum()

        t2.backward()

        assert t1.grad.data.tolist() == [1,1,1]

    def test_sum_with_grad (self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = t1.sum()

        t2.backward(Tensor(3))

        assert t1.grad.data.tolist() == [3,3,3]

class TestTensorAdd (unittest.TestCase):
    def test_simple_add (self):
        t1 = Tensor([1,2,3], requires_grad=True)
        t2 = Tensor([1,2,3], requires_grad=False)

        t3 = add(t1, t2)
        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-1,-2,-3]

    def test_broadcast_add1 (self):
        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True)
        t2 = Tensor([1,2,3], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([[-1,-1,-1], [-1,-1,-1]]))

        assert t1.grad.data.tolist() == [[-1,-1,-1], [-1,-1,-1]]
        assert t2.grad.data.tolist() == [-2,-2,-2]

    def test_broadcast_add2 (self):
        t1 = Tensor([[1,2,3], [4,5,6]], requires_grad=True)
        t2 = Tensor([[1,2,3]], requires_grad=True)

        t3 = add(t1, t2)
        t3.backward(Tensor([[-1,-1,-1], [-1,-1,-1]]))

        assert t1.grad.data.tolist() == [[-1,-1,-1], [-1,-1,-1]]
        assert t2.grad.data.tolist() == [[-2,-2,-2]]