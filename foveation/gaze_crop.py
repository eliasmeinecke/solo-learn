

import random
from PIL import Image

class GazeCenteredCrop:
    """
    Crop an N x N patch around a gaze point.
    Can be fixed, random, or externally provided.
    # N = 336, size = 540 for actual data
    """

    def __init__(
        self,
        crop_size=240,
        gaze=(358, 358),

    ):
        self.N = crop_size
        self.gaze = gaze

    def _correct_gaze(self, gaze, N, img_size):
    
        x_g, y_g = gaze

        x_cor = x_g - max(0, x_g + N//2 - img_size) - min(0, x_g - N//2)
        y_cor = y_g - max(0, y_g + N//2 - img_size) - min(0, y_g - N//2)    
        return (x_cor, y_cor)


    def _sample_gaze(self, img_size):
        if self.gaze:
            return self.gaze
        else:
            raise ValueError(f"Gaze value {self.gaze} not properly set.")

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        assert w == h, "expected square images"

        gaze = self._sample_gaze(w)
        x_cor, y_cor = self._correct_gaze(gaze, self.N, img_size=w)

        left   = x_cor - self.N // 2
        upper  = y_cor - self.N // 2
        right  = x_cor + self.N // 2
        lower  = y_cor + self.N // 2

        return img.crop((left, upper, right, lower))




if __name__ == "__main__":
    pass

    