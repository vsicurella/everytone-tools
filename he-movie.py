import he as h
import numpy as np 
import os

def makeFrames(outDir, numFrames, framePrefix=None, triadic=False, intLimitRange=None, spreadRange=None, alphaRange=None, dpi=144):
    
    nRange = (100, 100) if intLimitRange is None else intLimitRange
    fN = np.linspace(nRange[0], nRange[1], numFrames).astype(np.uint32)

    sRange = (15, 15) if spreadRange is None else spreadRange
    fS = np.linspace(sRange[0], sRange[1], numFrames)

    aRange = (7, 7) if alphaRange is None else alphaRange
    fA = np.linspace(aRange[0], aRange[1], numFrames)

    outPath = outDir
    if framePrefix is not None:
        outPath += os.sep + framePrefix
    
    he = h.HarmonicEntropy(spread=12, N=1, limit=1200, alpha=7, he3=triadic, out=outPath, dpi=dpi, verbose=0)

    updateInterval = numFrames // 10

    for n in range(numFrames):
        he.update(N=fN[n], s=fS[n], a=fA[n])
        he.calculate()
        he.makePlot(True, False, filename=f'{framePrefix}_{n}')
        if n > 0 and n % updateInterval == 0:
            print(f"Wrote {n} frames")
    
    print("Done.")


