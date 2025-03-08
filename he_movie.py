import he as h
import numpy as np
import plot_movie

class HeMovie(plot_movie.PlotMovie):
    intLimitRange = (100, 100)
    spreadRange = (15, 15)
    alphaRange = (7, 7)

    triadic = False
    
    basisGradient = False
    basisWeightFrames = []
    
    model = None

    def __init__(self, **kwArgs):
        if "intLimitRange" in kwArgs:
            self.intLimitRange = kwArgs["intLimitRange"]
        if "spreadRange" in kwArgs:
            self.spreadRange = kwArgs["spreadRange"]
        if "alphaRange" in kwArgs:
            self.alphaRange = kwArgs["alphaRange"]
        if "triadic" in kwArgs:
            self.triadic = kwArgs["triadic"]

        super().__init__(**kwArgs)
    
    def _prepare_(self):
        self.fN = np.linspace(self.intLimitRange[0], self.intLimitRange[1], self.numFrames).astype(np.uint32)
        self.fS = np.linspace(self.spreadRange[0], self.spreadRange[1], self.numFrames)
        self.fA = np.linspace(self.alphaRange[0], self.alphaRange[1], self.numFrames)

        self.model = h.HarmonicEntropy(self.fS[0], self.fN[0], 1, 1200, self.fA[0], self.triadic)
        self.model.calculate()

        self.basisWeightFrames = []
        if self.basisGradient:
            base_weights = self.model.basis_weights
            sorted_indices = sorted(np.asarray(range(base_weights.shape[0])), key=lambda i: base_weights[i],reverse=True)

            weights_per_frame = int(np.ceil(base_weights.shape[0] / self.numFrames))
            num_weights = 0
            frame_weights = np.zeros(base_weights.shape[0])
            for f in range(self.numFrames):
                end_i = num_weights+weights_per_frame
                frame_weights_i = sorted_indices[num_weights:end_i]
                if len(frame_weights_i) == 0:
                    break
                
                for i in frame_weights_i:
                    frame_weights[i] += base_weights[i]
                self.basisWeightFrames.append(frame_weights.copy())
                
                num_weights = end_i

        super()._prepare_()

    def makeFrame(self, i, **kwArgs):
        if self.basisGradient:
            self.model._distribute_basis(None, self.basisWeightFrames[i])
        else:
            self.model.update({'N': self.fN[i], 's': self.fS[i], 'a': self.fA[i]})
        
        data = self.model.getEntropy()
        self.axis.imshow(data, cmap="inferno", aspect="equal", origin="lower")




directory = "he_movie_plot_movie"
filename = "basis_grad_test_reverse"

heMovie = HeMovie(numFrames=100, triadic=True, basisGradient=True)
heMovie.makeMovie()
heMovie.saveMovie(directory, filename, frameRateMs=150)
print("Done.")

