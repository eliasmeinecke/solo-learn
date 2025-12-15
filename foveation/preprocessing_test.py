
from PIL import Image
import torchvision.transforms as T

from gaze_crop import GazeCenteredCrop



def test_transform(transform):
    # Test if transform is created correctly
    
    img = Image.open("visualisation_data/skating_panda.jpg").convert("RGB")

    x = transform(img)

    print(type(x))
    print(x.shape)




if __name__ == "__main__":
    # Define transform to be tested here
    transform = T.Compose([
            GazeCenteredCrop(240, gaze=(358, 358)),
            T.ToTensor()
        ])
    test_transform(transform)