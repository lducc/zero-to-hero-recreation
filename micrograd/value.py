import numpy as np
import math

class Value:
    """ Hello my name is Anh Duc"""
    def __init__(self, val, _children=(), _op = '', label = ''):
        self.val = val
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        #This is a commnent
        return f"Value(val = {self.val})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self.val + other.val
        obj = Value(out, _children=(self, other), _op='+')
        #This is alsoooooooooooooo a commnent

        def _backward():
            self.grad += obj.grad * 1.0
            other.grad += obj.grad * 1.0

        obj._backward = _backward
        return obj

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = self.val * other.val
        obj = Value(out, _children=(self, other), _op="*")

        def _backward():
            self.grad += obj.grad * other.val
            other.grad += obj.grad * self.val

        obj._backward = _backward
        return obj

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** - 1

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = self.val ** other.val
        obj = Value(out, _children=(self, other), _op='^')

        def _backward():
            #a ** b ->
            if self.val != 0:
                self.grad += obj.grad * other.val * self.val ** (other.val - 1) #a ** b -> b * a ** (b - 1)
            else:
                self.grad += 0

            if self.val > 0:
                other.grad += obj.grad * out * math.log(self.val)
            else:
                other.grad += 0

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
        self.grad = 1.0
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


    def exp(self):
        # e to the power of x
        out = math.exp(self.val)
        obj = Value(out, _children=(self, ), _op='exp')

        def _backward():
            self.grad += obj.grad * obj.val

        obj._backward = _backward

        return obj

