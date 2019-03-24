#!/usr/bin/env python3


import unittest
import sys
sys.path.append("..")

from singleneuron import Neuron

import numpy as np

np.random.seed(1)

class NeuronTest(unittest.TestCase):
    
    
    def test_constructor(self):
        """Neuron gets created with random weights in (-1, 1) range"""
        for i in range(1,10):
            dut = Neuron(i)
            
            self.assertTrue( (np.abs(dut.coeffs) - 0.5 < 0.5).all(), dut.coeffs)
        
            self.assertEqual((i,1), dut.coeffs.shape)
            self.assertEqual(None, dut.last_inputs)
            self.assertEqual(None, dut.last_output)

    def test_forward_single(self):
        """test forward with just one set of inputs"""
        dut = Neuron(3)
        # replace the weigths
        dut.coeffs = np.array([1,2,3]).reshape((3,1))
        self.assertEqual(dut.coeffs.shape, (3, 1))

        output = dut.forward(np.array([0,0,0]))
        self.assertEqual((1,1), output.shape)
        self.assertEqual(0.5, output)
        self.assertTrue((np.array([0,0,0]) == dut.last_inputs).all())
        self.assertEqual(output, dut.last_output)

        output = dut.forward(np.array([3, -3, 1]))
        self.assertEqual(0.5, output)

    def test_backward_single(self):

        dut = Neuron(3)

        inputs = np.array([3, -3, 1])
        output = dut.forward(inputs)
        self.assertEqual((1, 3), dut.last_inputs.shape)
        dut.backward(1)
        self.assertEqual((3, 1), dut.coeffs.shape)

    def test_forward_multiple(self):
        """test forward processing simulaneously multiple sets of inputs"""
        sz = 3
        mul = 10
        dut = Neuron(sz)
        # replace the weigths
        dut.coeffs = np.array([1,2,3]).reshape((sz, 1))
        self.assertEqual(dut.coeffs.shape, (sz, 1))

        output = dut.forward(np.zeros((mul, sz)))
        self.assertEqual((mul, 1), output.shape)
        self.assertTrue((0.5*np.ones((mul, 1)) == output).all())

    def test_backward_mulitple(self):
        """test backward propagation with multiple inputs"""

        sz = 3
        mul = 10
        dut = Neuron(sz)

        inputs = np.zeros((mul, sz))
        output = dut.forward(inputs)
        self.assertEqual((mul, sz), dut.last_inputs.shape)
        dut.backward(output)
        self.assertEqual((3, 1), dut.coeffs.shape)
        
        


        
    

if __name__ == "__main__":
    unittest.main()
        
