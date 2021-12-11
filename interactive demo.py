import tkinter as tk
from tkinter.constants import NO
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from threading import Thread
from outerbilliards import *
from geometry import *
import utils

class InteractiveDemo(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("500x500")
        self.title("Polygon Outer Billiards Demo")

        self.vertices = utils.polygonVertices(5)
        self.structure = LineSet()

        self.fig = Figure(figsize = (5, 5), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        self.structurePlot, = self.structure.plot(
            plot=self.plot,
            setAspect=False
        )

        self.polygonPlot, = LineSet.connect(self.vertices).plot(
            color="r",
            size=1,
            plot=self.plot,
            setAspect=False
        )

        self.redrawStructure()

        self.canvasWidget = self.canvas.get_tk_widget()
        self.canvasWidget.pack()
        self.canvasWidget.bind("<Button-1>", self.redrawStructure)
    
    def run(self):
        self.mainloop()
    
    def redrawStructure(self, *args):
        self.structurePlot.set_xdata(np.random.random((10000, 1)))
        self.structurePlot.set_ydata(np.random.random((10000, 1)))
        self.canvas.draw()


app = InteractiveDemo()
app.run()

"""
def plot():
    global plot1
    global canvas
    global line

    y = np.random.random((100, 1))

    if plot1 is None:
        fig = Figure(figsize = (5, 5), dpi=100)
        plot1 = fig.add_subplot(111)
        line, = plot1.plot(y)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()

        canvas.get_tk_widget().pack()
    else:
        line.set_ydata(y)
        canvas.draw()

window = tk.Tk()

window.geometry("500x500")

plot_button = tk.Button(master=window,
                    command=plot,
                    height=2,
                    width=10,
                    text="Plot")

plot_button.pack()

window.mainloop()
"""