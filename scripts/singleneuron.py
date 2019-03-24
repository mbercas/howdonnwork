#!/usr/bin/env python3
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, coeffs_len):
        """The legth of the vector coeffs defines the number of inputs.
        Initializes the coefficients to a random number between (-1,1)"""
        self.coeffs = 2*np.random.random((coeffs_len, 1)) - 1
        self.last_inputs = None
        self.last_output = None
    
    def forward(self, inputs):
        """Multiply the inputs by the coefficients and pass through the sigmoid function"""

        if inputs.ndim == 1:
            inputs = inputs.reshape((1, inputs.shape[0]))
            
        self.last_inputs = inputs
        self.last_output = sigmoid(np.dot(inputs, self.coeffs))
        return self.last_output
    
    def backward(self, error):
        """Update the coefficients based on how much error at the output, use the derivative
        of the sigmoid function to decide how to jump in the coefficient calculation"""
        delta = error * sigmoid_deriv(self.last_output)
        self.coeffs += np.dot(self.last_inputs.T, delta) #(self.last_inputs * delta).T



if __name__ == "__main__":
    coeffs_sz = 3
    N = Neuron(coeffs_sz)
    # y = 0.1*x1 + 0.3*x2 -0.1*x3

    def model(x):
        coeffs = np.array([0.1, 0.01, -0.1])
        return 1*(np.inner(x, coeffs) > 0)

    #assert( model(np.array([[1,2,3]])) - 0.4 < 0.001 )

    sz = 1500
    reps = 2000
    X = np.random.random((sz,3))-1
    Y = model(X).reshape((sz,1))



    model_coeffs = np.zeros((reps, N.coeffs.shape[0]))
    output_error = np.zeros(reps)

    model_coeffs[0] = N.coeffs.reshape((3,))
    for i in range(1, reps):
        output = N.forward(X)
        l_error = Y - output
        output_error[i] = np.linalg.norm(l_error,ord=2)
        #print(l_error)
        N.backward(l_error)
        model_coeffs[i] = N.coeffs.reshape((3,))
    

    print(output)
    print(Y)
    print(N.coeffs/sz)

    plt.subplot(211)
    plt.plot(model_coeffs); plt.xlabel("iterations"); plt.legend(["x1", "x2", "x3"])
    plt.subplot(212)
    plt.plot(output_error)
    plt.show()
