import numpy as np
import csv
import math
import random


def pointColor(output):
    if output == 0:
        return "darkorange"
    if output == 1:
        return "seagreen"


def load_inputs(example):
    x = []
    y = []
    with open(f"./examples/{example}/inputs.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                x.append(int(row[0]))
                y.append(int(row[1]))
            line_count += 1
    return x, y


def load_outputs(example):
    outputs = []
    with open(f"./examples/{example}/outputs.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                outputs.append(int(row[0]))
            line_count += 1
    return outputs


def load_from_file(example):
    inputs = []
    outputs = []
    with open(f"./examples/example_{example}.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            inputs.append([float(row["x0"]), float(row["x1"]), float(row["x2"])])
            outputs.append(int(row["d1"]))
            line_count += 1

    return inputs, outputs


def createWeights(neurons, hidden):
    w = []
    if hidden == True:
        for i in range(neurons):
            w.append([random.random(), random.random(), random.random()])
        return np.array(w)
    for i in range(neurons + 1):
        w.append(random.random())
    return np.array(w)


def guess(inputs, weights):
    return sigmoidf(np.dot(weights.T, inputs))
    # for i in range(len(w)):
    #     w += sigmoid * lr  * error * inputs[i]
    # return w


def sigmoidf(x, derivative=False):
    if derivative == True:
        return x * (1 - x)
    return 1 / (1 + math.exp(-x))


def guessY(x, w):
    m = w[1] / w[2]
    b = w[0] / w[2]
    return -m * x - b


def feedForward(hiddenWeights, outputWeights, hiddenY, point):
    outputY = 0
    for i in range(1, len(hiddenY)):
        hiddenY[i] = guess(inputs=point, weights=hiddenWeights[i - 1])
    outputY = guess(inputs=hiddenY, weights=outputWeights)

    return hiddenY, outputY


def backPropagation(outputY, point, pointsY, hiddenY, hiddenWeights, outputWeights):
    outputGradient = getLocalGradient(outputY, pointsY, True)
    for i in range(len(outputWeights)):
        outputWeights[i] += outputGradient * hiddenY[i]  # * lr
    outputWeightsSum = sum(outputWeights)
    for i in range(len(hiddenWeights)):
        hiddenGradient = (
            getLocalGradient(hiddenY[i]) * outputWeightsSum * outputGradient
        )
        for j in range(len(hiddenWeights[i])):
            hiddenWeights[i][j] += hiddenGradient * point[j]

    return hiddenWeights, outputWeights


def getLocalGradient(y, pointY=0, outputLayer=False):
    sigmoid = sigmoidf(y, True)
    if outputLayer == True:
        error = pointY - y
        return sigmoid * error
    return sigmoid

    # pass
