import numpy as np 
import os
from PIL import Image
import matplotlib.pyplot as plt

class PlotMovie:

    numFrames = 240

    figure = None
    axis = None

    def __init__(self, **kwArgs):
        if "numFrames" in kwArgs:
            self.numFrames = kwArgs["numFrames"]
        if "figsize" in kwArgs:
            self.figsize = kwArgs["figsize"]
        else:
            self.figsize = (1000, 1000)

    def _prepare_(self):
        self.figure = plt.figure(figsize=self.figsize, dpi=1)
        self.axis = plt.gca()
        self.axis.set_axis_off()
        plt.tight_layout(pad=0)

        self.frames = []

        self.updateInterval = self.numFrames // 10

    def _renderFrame_(self):
        self.figure.canvas.draw()
        rgb_buffer = self.figure.canvas.tostring_rgb()
        frame = Image.frombytes("RGB", self.figure.canvas.get_width_height(), rgb_buffer)
        return frame

    def makeFrame(self, i, **kwArgs):
        pass
    
    def makeMovie(self):
        self._prepare_()

        for n in range(self.numFrames):

            self.makeFrame(n)
            frame = self._renderFrame_()
            self.frames.append(frame)

            if n > 0 and n % self.updateInterval == 0:
                print(f"Rendered {n} frames")
            
        return True
    
    def saveMovie(self, outdir, filename, frameRateMs=100):
        filepath = os.path.join(outdir, filename + ".gif")
        frame = self.frames[0]
        with frame.copy() as out:
            out.save(
                filepath,
                save_all=True,
                append_images=self.frames[1:],
                optimize=True,
                duration=frameRateMs,
                loop=0
            )
