from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.core.numeric import extend_all
from Functions import *

# funcion contorno


class Window:
    def __init__(self, window):
        self.window = window
        self.window.title("Práctica 4")
        self.window.geometry("750x500")

        # Values Init
        self.points = []
        self.pointsOutputs = []
        self.hiddenWeights = np.zeros((0, 3))
        self.hiddenOutputs = None
        self.outputWeights = np.zeros(0)
        self.outputOutput = 0
        self.epoch = 0

        # Entries
        self.entryLabels = ["Neurons: ", "Epochs: ", "Learning Rate: ", "Example: "]
        self.entries: Entry = []

        # Buttons
        self.loadBtn: Button
        self.startBtn: Button
        self.cleanBtn: Button

        # Labels
        self.epochLabel: Label
        self.MSELabel: Label

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

        for i in range(len(self.entryLabels)):
            Label(master=upperFrame, text=self.entryLabels[i]).grid(
                row=i, column=0, sticky=E
            )
            self.entries.append(Entry(master=upperFrame, width=6))
            self.entries[i].grid(row=i, column=1, sticky=W)

        Label(upperFrame, text="", width=12).grid(row=5, column=0)

        middleFrame = Frame(actionFrame)
        middleFrame.grid(row=1, column=0)

        self.loadBtn = Button(
            master=middleFrame, text="Load Example", command=self.load_example, width=15
        )
        self.loadBtn.grid(row=0, column=0)

        self.startBtn = Button(
            master=middleFrame, text="Start", command=self.start, width=15
        )
        self.startBtn.grid(row=1, column=0)

        self.cleanBtn = Button(
            master=middleFrame, text="Clean", command=self.clean, width=15
        )
        self.cleanBtn.grid(row=2, column=0)

        Label(master=middleFrame, text="", width=12).grid(row=4, column=0)

        lowerFrame = Frame(actionFrame)
        lowerFrame.grid(row=2, column=0)

        Label(lowerFrame, text="Epoch: ").grid(row=1, column=0, rowspan=2)
        self.epochLabel = Label(lowerFrame, text="0")
        self.epochLabel.grid(row=1, column=1, rowspan=2, sticky=W)

        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.graph = self.figure.add_subplot(111)
        self.configGraph()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().grid(row=0, column=2)

    def configGraph(self):
        self.graph.cla()  # Clears graph
        self.graph.grid()  # Adds grid to graph
        # Sets graph limits
        self.graph.set_xlim([self.graphLimits[0], self.graphLimits[1]])
        self.graph.set_ylim([self.graphLimits[0], self.graphLimits[1]])
        # Draw origin lines
        self.graph.axhline(y=0, color="k")
        self.graph.axvline(x=0, color="k")

    def start(self):
        neurons = int(self.entries[0].get())
        totalEpochs = int(self.entries[1].get())

        self.hiddenWeights = createWeights(neurons, True)
        self.hiddenOutputs = np.zeros(neurons + 1)
        self.outputWeights = createWeights(neurons, False)
        self.hiddenOutputs[0] = 1

        if self.entries[1].get() == "":
            self.epoch = 100
        else:
            self.epoch = totalEpochs
        epoch = 0

        while epoch <= 1:
            self.window.update()
            self.epochLabel.config(text=epoch)
            # self.configGraph()

            for i in range(len(self.points)):
                self.outputOutput = feedForward(
                    hiddenWeights=hiddenWeights[j], 
                    outputWeights=outputWeights, 
                    point=self.points[i]
                )

        #     self.canvas.draw()
        #     epoch += 1

    def clean(self):
        self.points = []
        self.pointsOutputs = []
        self.configGraph()
        self.graph.plot()
        self.canvas.draw()
        # self.MSELabel.config(text="")
        self.epochLabel.config(text="0")

    def load_example(self):
        example = int(self.entries[3].get())

        if example == 0 or example == "":
            return

        self.clean()

        inputs, outputs = load_from_file(example=example)
        self.points = np.array(inputs)
        self.pointsOutputs = np.array(outputs)

        for i in range(len(inputs)):
            self.plot(inputs[i][1], inputs[i][2], outputs[i])

        self.canvas.draw()

    def plot(self, x, y, desire):
        color = pointColor(desire)
        self.graph.plot(x, y, marker="o", color=color)


window = Tk()
app = Window(window)
window.mainloop()
