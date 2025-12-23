from PIL import Image
import matplotlib.pyplot as plt
from gaze_crop import GazeCenteredCrop


img = Image.open("visualisation_data/skating_panda.jpg").convert("RGB")


def viz_crop(img)
    crop = GazeCenteredCrop(
        crop_size=240,
        gaze=(358, 358)
    )


    out = crop(img)


    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Gaze Crop")
    plt.imshow(out)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plots/crop_example.png", dpi=200)
    print("Saved crop_example.png")    

    