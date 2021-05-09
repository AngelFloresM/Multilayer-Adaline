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


def guessY(x, w):
    m = w[1] / w[2]
    b = w[0] / w[2]
    return -m * x - b
