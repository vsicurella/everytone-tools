import he as h
import numpy as np 
import os
from PIL import Image
import matplotlib.pyplot as plt

def makeFrames(numFrames, triadic=False, **kwArgs):
    
    nRange = (100, 100) if "intLimitRange" not in kwArgs else kwArgs["intLimitRange"]
    fN = np.linspace(nRange[0], nRange[1], numFrames).astype(np.uint32)

    sRange = (15, 15) if "spreadRange" not in kwArgs else kwArgs["spreadRange"]
    fS = np.linspace(sRange[0], sRange[1], numFrames)

    aRange = (7, 7) if "alphaRange" not in kwArgs else kwArgs["alphaRange"]
    fA = np.linspace(aRange[0], aRange[1], numFrames)

    # outDir = None if "outdir" not in kwArgs else kwArgs["outDir"]
    # framePrefix = None if "framePrefix" not in kwArgs else kwArgs["framePrefix"]

    he = h.HarmonicEntropy(spread=12, N=50, limit=1200, alpha=7, he3=triadic,  verbose=0)
    he.calculate()

    basisGradient = False
    basisWeightFrames = []
    if "basisGradient" in kwArgs:
        basisGradient = True
        base_weights = he.basis_weights
        sorted_indices = sorted(np.asarray(range(base_weights.shape[0])), key=lambda i: base_weights[i],reverse=True)

        weights_per_frame = int(np.ceil(base_weights.shape[0] / numFrames))
        num_weights = 0
        frame_weights = np.zeros(base_weights.shape[0])
        for f in range(numFrames):
            end_i = num_weights+weights_per_frame
            frame_weights_i = sorted_indices[num_weights:end_i]
            if len(frame_weights_i) == 0:
                break
            
            for i in frame_weights_i:
                frame_weights[i] += base_weights[i]
            basisWeightFrames.append(frame_weights.copy())
            
            num_weights = end_i

    updateInterval = numFrames // 10

    figsize = (1000, 1000)
    fig = plt.figure(figsize=figsize, dpi=1)
    ax = plt.gca()
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    frames = []
    # saveFrames = False
    for n in range(numFrames):

        if basisGradient:
            he._distribute_basis(None, basisWeightFrames[n])
        else:
            he.update({'N': fN[n], 's': fS[n], 'a': fA[n]})
        
        data = he.getEntropy()
        
        ax.imshow(data, cmap="inferno", aspect="equal", origin="lower")
        fig.canvas.draw()
        rgb_buffer = fig.canvas.tostring_rgb()
        frame = Image.frombytes("RGB", fig.canvas.get_width_height(), rgb_buffer)

        # if saveFrames:
        #     filename = f'{framePrefix}_{n}'
        #     # todo

        frames.append(frame)

        if n > 0 and n % updateInterval == 0:
            print(f"Rendered {n} frames")

    return frames


def saveFrames(outdir, filename, frames, frameRateMs=100):
    filepath = os.path.join(outdir, filename + ".gif")
    frame = frames[0]
    with frame.copy() as out:
        out.save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=frameRateMs,
            loop=0
        )
    

directory = "he_movie2"
filename = "basis_grad_test_reverse"

frames = makeFrames(100, True, basisGradient=True)
saveFrames(directory, filename, frames, frameRateMs=150)
print("Done.")
