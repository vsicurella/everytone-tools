import numpy as np
import plot_movie

from octave_burst import *

class BurstMovie(plot_movie.PlotMovie):
    resolutionRange = (256, 256)
    centsWindowRange = (1e-3, 1.5)
    edoRange = (5, 3130)
    optimized = True

    def __init__(self, **kwArgs):
        if "resolutionRange" in kwArgs:
            self.resolutionRange = kwArgs["resolutionRange"]
        if "centsWindowRange" in kwArgs:
            self.centsWindowRange = kwArgs["centsWindowRange"]
        if "alphaRange" in kwArgs:
            self.edoRange = kwArgs["edoRange"]
        if "optimized" in kwArgs:
            self.optimized = kwArgs["optimized"]

        super().__init__(**kwArgs)
    
    def _prepare_(self):
        self.frame_res = np.linspace(self.resolutionRange[0], self.resolutionRange[1], self.numFrames).astype(np.uint32)
        self.frame_cw = np.linspace(self.centsWindowRange[0], self.centsWindowRange[1], self.numFrames)
        self.frame_edo = np.linspace(self.edoRange[0], self.edoRange[1], self.numFrames).astype(np.uint32)

        super()._prepare_()

    def makeFrame(self, i, **kwArgs):
        data = make_burst_gradient(self.frame_res[i], 1200, self.frame_cw[i], self.frame_edo[i], self.optimized)
        # self.axis.imshow(data, cmap="inferno", aspect="equal", origin="lower")
        self.axis.imshow(data)



directory = "burst_movie"
filename = "burst_movie2"

burstMovie = BurstMovie(numFrames=24, figsize=(512, 512), edoRange=(313, 939), centsWindowRange=(0.6, 1e-2))
burstMovie.makeMovie()
burstMovie.saveMovie(directory, filename, frameRateMs=200)
print("Done.")

