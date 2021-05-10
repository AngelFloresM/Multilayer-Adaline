from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import csv

from tkinter import *
from tkinter import ttk

class Window:
    def __init__(self, window):
        self.window = window
        self.window.title("Práctica 4")
        self.window.geometry("800x500")

        # Values Init
        self.points = []
        self.classes = ["0", "1"]

        # Buttons
        self.createBtn = None
        self.cleanBtn = None

        self.figure = None
        self.graph = None
        self.canvas = None
        self.graphLimits = [-2, 2]

        self.startUI()

    def startUI(self):

        actionFrame = Frame(self.window)
        actionFrame.grid(row=0, column=0, padx=20, ipady=10)

        upperFrame = Frame(actionFrame)
        upperFrame.grid(row=0, column=0)

        Label(upperFrame, text="Class: ").grid(row=0, column=0, sticky=E)
        self.classBox = ttk.Combobox(upperFrame, values=self.classes, width=10)
        self.classBox.grid(row=0, column=1)
        self.classBox.current(0)

        Label(upperFrame, text="", width=12).grid(row=1, column=0)

        middleFrame = Frame(actionFrame)
        middleFrame.grid(row=1, column=0)

        self.createBtn = Button(
            middleFrame, text="Create File", command=self.createFile, width=10
        )
        self.createBtn.grid(row=0, column=0)

        self.cleanBtn = Button(middleFrame, text="Clean", command=self.clean, width=10)
        self.cleanBtn.grid(row=0, column=1)

        Label(middleFrame, text="", width=12).grid(row=1, column=0)

        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.graph = self.figure.add_subplot(111)
        self.configGraph()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().grid(row=0, column=2)

        cid = self.figure.canvas.mpl_connect("button_press_event", self.onClick)

        self.configGraph()

    def configGraph(self):
        self.graph.cla()  # Clears graph
        self.graph.grid()  # Adds grid to graph
        # Sets graph limits
        self.graph.set_xlim([self.graphLimits[0], self.graphLimits[1]])
        self.graph.set_ylim([self.graphLimits[0], self.graphLimits[1]])
        # Draw origin lines
        self.graph.axhline(y=0, color="k")
        self.graph.axvline(x=0, color="k")

    def onClick(self, event: Event):
        classValue = int(self.classBox.current())

        self.points.append([float(event.xdata), float(event.ydata), classValue])

        self.plot(float(event.xdata), float(event.ydata), classValue)

        self.canvas.draw()

    def clean(self):
        self.points = []
        self.configGraph()
        self.graph.plot()
        self.canvas.draw()

    def plot(self, x, y, desire):
        if(desire==0):
            self.graph.plot(x, y, marker="o", color="green")
        else:
            self.graph.plot(x, y, marker="o", color="red")

    def createFile(self):
        # Change example number to not overwrite the examples you already have created
        with open("examples/example_1.csv", 'w', newline='') as fs:
            fieldnames = ["x0", "x1", "x2", "d1"]
            writer = csv.DictWriter(fs, fieldnames=fieldnames)

            writer.writeheader()
            for point in self.points:
                writer.writerow({"x0": 1, "x1": point[0], "x2": point[1], "d1": point[2]})



window = Tk()
app = Window(window)
window.mainloop()
