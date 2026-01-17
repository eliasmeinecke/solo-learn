
import numpy as np
from PIL import Image
import pandas as pd
import math
from foveation.methods.base import Foveation

"""
Implementation of:
"A NEW FOVEAL CARTESIAN GEOMETRY APPROACH USED FOR OBJECT TRACKING" - José Martínez, Leopoldo Altamirano
"""


class FovealCartesianGeometry(Foveation):
    def __init__(self, p0=30, pmax=200, nR=60):
        """
        p0   : radius of fovea (in pixels)
        pmax : ?
        nR   : number of rings
        """
        self.p0 = p0
        self.pmax = pmax
        self.nR = nR

        self.prepare_geometry()
        
    def prepare_geometry(self):
        
        self.a = math.exp(math.log((self.pmax / self.p0)) / self.nR)
        self.r = 0

        ps = []
        
        self.gammas = [self.p0]
        self.rings = []

        for i in range(self.nR):
            ps.append(math.floor(self.p0 * math.pow(self.a, i))) 
            if ps[i] != self.gammas[self.r]:
                self.r += 1
                self.gammas.append(ps[i])
            self.rings.append(self.r) 
        
        self.fovea_size = 2*(self.p0+self.r)+1
        
        self.fcx = self.fcy = self.fovea_size//2
    
    def mapping(self, x, y, x0=0, y0=0):
        
        x -= x0
        y -= y0

        p = max(abs(x), abs(y)) # has to be p instead of r, right?
        
        if p <= self.p0:
            xp = x + self.fcx
            yp = y + self.fcy
        else:
            eta = int(math.floor(math.log((p/self.p0), self.a))) # also has to be p instead of r, right?
            eta = min(eta, len(self.rings) - 1) # added because indexerror occured :( fix this?
            delta_roh_eta = (self.p0 + self.rings[eta]) / math.floor(self.p0 * math.pow(self.a, eta))
            xp = int(math.floor(x * delta_roh_eta + self.fcx))
            yp = int(math.floor(y * delta_roh_eta + self.fcy))

        return xp, yp 

    def inverse_mapping(self, xp, yp, x0, y0):

        xp -= self.fcx
        yp -= self.fcy

        p = max(abs(xp), abs(yp))

        if p <= self.p0:
            x = xp + x0
            y = yp + y0
        else:
            gamma = self.gammas[min(p - self.p0, len(self.gammas)-1)]
            delta_roh_gamma = gamma / p
            x = int(math.floor(xp * delta_roh_gamma + x0))
            y = int(math.floor(yp * delta_roh_gamma + y0))
        
        return x, y
    
    
    def __call__(self, img: Image.Image, annot: pd.Series) -> Image.Image:
        
        img_np = np.array(img)
        
        H, W, C = img_np.shape

        
        # change to gaze_center (annot.gaze_loc_x / y) and add resize!
        x0 = W // 2
        y0 = H // 2
        
        out = np.zeros((self.fovea_size, self.fovea_size, C), dtype=img_np.dtype)

        for yp in range(self.fovea_size):
            for xp in range(self.fovea_size):
                x, y = self.inverse_mapping(xp, yp, x0, y0)

                if 0 <= x < W and 0 <= y < H:
                    out[yp, xp] = img_np[y, x]
                # outside periphery - remain black

        out = Image.fromarray(out)
        
        return out.resize(
            (540, 540),
            resample=Image.BILINEAR
        )
