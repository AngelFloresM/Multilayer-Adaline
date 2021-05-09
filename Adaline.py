import numpy as np
import math


class Adaline:
    def __init__(self, lr, inputs=3, outputLayer=False):
        self.lr = lr
        self.inputs = inputs
        self.outputLayer = outputLayer
        self.w = np.zeros(inputs)
        for i in range(self.inputs):
            self.w[i] = np.random.uniform(-2, 2)
        self.y = np.zeros(0)
        self.localGradient = 0

    def guess(self, p):
        return self.sigmoid(np.dot(self.w.T, p))

    def sigmoid(self, x, derivative=False):
        if derivative == True:
            return x * (1 - x)
        return 1 / (1 + math.exp(-x))

    def getOutput(self, p):
        self.y = self.guess(p)

    def backPropagation(self, prevLayerY=None, nextLayer=None, pointY=None):
        if self.outputLayer == True:
            self.localGradient = self.localGradientFunc(pointY=pointY)
            for i in range(self.inputs):
                self.w[i] += self.localGradient * prevLayerY[i] * self.lr
        else:
            self.localGradient = self.localGradientFunc(
                pointY=pointY, nextLayer=nextLayer
            )
            for i in range(self.inputs):
                self.w[i] += self.localGradient * prevLayerY[i] * self.lr

    def localGradientFunc(self, pointY, nextLayer=None):
        sigmoid = self.sigmoid(self.y, True)
        if self.outputLayer == True:
            error = pointY - self.y
            return sigmoid * error
        wSum = sum(nextLayer.w)
        return sigmoid * wSum * nextLayer.localGradient
