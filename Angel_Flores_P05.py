import sys
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# sys.path.append('C:\Users\angel\OneDrive\Documentos\CUCEI\SSPAI2\P05')

from Functions import *
from Adaline import *

# funcion contorno


class Window:
    def __init__(self, window):
        self.window = window
        self.window.title("Práctica 4")
        self.window.geometry("750x500")

        self.colors = ("red", "blue")
        self.cmap = ListedColormap(self.colors[: len(np.unique([0, 1]))])

        # Values Init
        self.points = []
        self.pointsY = []

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
        self.limits = [-2, 2]

        self.x = np.linspace(self.limits[0], self.limits[1], 50)
        self.y = np.linspace(self.limits[0], self.limits[1], 50)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.inputs = np.array(
            [np.ones(len(self.xx.ravel())), self.xx.ravel(), self.yy.ravel()]
        ).T
        self.outputs = np.zeros(len(self.inputs))

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
        self.graph.set_xlim([self.limits[0], self.limits[1]])
        self.graph.set_ylim([self.limits[0], self.limits[1]])
        # Draw origin lines
        self.graph.axhline(y=0, color="k")
        self.graph.axvline(x=0, color="k")

    def start(self):
        neurons = int(self.entries[0].get())
        totalEpochs = int(self.entries[1].get())
        lr = float(self.entries[2].get())
        epoch = 0
        hiddenLayer = np.array([])

        outputLayer = Adaline(lr=lr, inputs=neurons + 1, outputLayer=True)

        for i in range(neurons):
            hiddenLayer = np.append(hiddenLayer, Adaline(lr=lr))

        grid = np.zeros((neurons, len(self.inputs)))

        while epoch <= totalEpochs:
            self.window.update()
            self.epochLabel.config(text=epoch)
            self.configGraph()

            for i in range(len(self.points)):
                [layer.getOutput(self.points[i]) for layer in hiddenLayer]

                outputLayer.getOutput(np.array([1] + [n.y for n in hiddenLayer]))
                outputLayer.backPropagation(
                    prevLayerY=[1] + [n.y for n in hiddenLayer], pointY=self.pointsY[i]
                )

                [
                    layer.backPropagation(
                        prevLayerY=self.points[i], nextLayer=outputLayer
                    )
                    for layer in hiddenLayer
                ]

            for neuron in hiddenLayer:
                self.graph.plot(
                    [-2, 2], [guessY(-2, neuron.w), guessY(2, neuron.w)], c="slategrey"
                )

            for i in range(len(self.points)):
                self.graph.plot(
                    self.points[i][1],
                    self.points[i][2],
                    marker="o",
                    c=pointColor(self.pointsY[i]),
                )

            epoch += 1

            for j in range(neurons):
                grid[j] = hiddenLayer[j].guess(self.inputs)
            self.outputs = [
                outputLayer.guess(np.concatenate((np.array([1]), [g]), axis=None))
                for g in grid.T
            ]
            self.outputs = np.array(self.outputs)

            self.graph.contourf(
                self.xx, self.yy, self.outputs.reshape(self.xx.shape), cmap="magma"
            )

            self.canvas.draw()

    def clean(self):
        self.points = []
        self.pointsY = []
        self.configGraph()
        self.graph.plot()
        self.canvas.draw()
        # self.MSELabel.config(text="")
        self.epochLabel.config(text="0")

    def load_example(self):
        self.clean()
        example = int(self.entries[3].get())

        inputs, outputs = load_from_file(example=example)
        self.points = np.array(inputs)
        self.pointsY = np.array(outputs)

        for i in range(len(self.points)):
            self.graph.plot(
                self.points[i][1],
                self.points[i][2],
                marker="o",
                c=pointColor(self.pointsY[i]),
            )

        self.canvas.draw()


window = Tk()
app = Window(window)
window.mainloop()
