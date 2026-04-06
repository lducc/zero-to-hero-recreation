# import math
# import numpy as np

# class Value:
#     def __init__(self, val, _children=(), _op='', label = ''):
#         self.val = val
#         self.grad = 0.0
#         self._backward = lambda: None
#         self._prev = set(_children)
#         self._op = _op
#         self.label = label

#     def __repr__(self):
#         return f"Value(val={self.val})"

#     def __add__(self, other):
#         output = self.val + other.val
#         out = Value(output, (self, other), '+')

#         def _backward():
#             self.grad = out.grad * 1.0
#             other.grad = out.grad * 1.0

#         out._backward = _backward
#         return out

#     def __mul__(self, other):
#         output = self.val * other.val
#         out = Value(output, (self, other), '*')

#         def _backward():
#             self.grad = out.grad * other.val
#             other.grad = out.grad * self.val

#         out._backward = _backward
#         return out

#     def tanh(self):
#         output = (math.exp(self.val) - math.exp(-self.val)) / (math.exp(self.val) + math.exp(-self.val))
#         out = Value(output, (self, ), 'tanh')

#         def _backward():
#             self.grad = (1 - output ** 2) * out.grad

#         out._backward = _backward
#         return out

# if __name__ == "__main__":
#     a = Value(2.0)
#     b = Value(-3.0)
#     c = Value(10.0)

#     d = a * b + c
#     print(d._prev)
#     print(d._op)

from turtle import back

import numpy as np
import math

class Value:
    def __init__(self, val, _children=(), _op = '', label = ''):
        self.val = val
        self.grad = 0
        self._prev = set(_children)
        self._op = ''
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(val = {self.val})"

    def __add__(self, other):
        out = self.val + other.val
        obj = Value(out, _children=(self, other), _op='+')

        def _backward():
            self.grad += obj.grad * 1.0
            other.grad += obj.grad * 1.0

        obj._backward = _backward
        return obj

    def __mul__(self, other):
        out = self.val * other.val
        obj = Value(out, _children=(self, other), _op="*")

        def _backward():
            self.grad += obj.grad * other.val
            other.grad += obj.grad * self.val

        obj._backward = _backward
        return obj

    def tanh(self):
        x = self.val
        out = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        obj = Value(out, _children=(self,), _op="tanh")

        def _backward():
            self.grad += obj.grad * (1 - obj.val ** 2)

        obj._backward = _backward
        return obj


    def backward(self):
        topo = []
        visited = set()

        def visit_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit_topo(child)

                topo.append(node)

        visit_topo(self)

        for node in reversed(topo):
            node._backward()

