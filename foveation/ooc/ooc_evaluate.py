import argparse
import torch
import torchvision.transforms.v2 as v2

from foveation.factory import setup_exact_foveation
from foveation.ooc.ooc_utils import load_data, load_model, IdentityFoveation


def evaluate(model, loader, foveation, T_post, device):

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():

        for img, label, gaze in loader:

            img = img.to(device)
            label = label.to(device)
            gaze = gaze.to(device)

            img = foveation(img, gaze)
            img = T_post(img)

            logits = model(img)

            pred = logits.argmax(dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

    return correct / total


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="base") # has to be a valid foveation type f.e. cm-light
    parser.add_argument("--dataset", type=str, default="original") # original, inpainted, object, ooc 
    
    args = parser.parse_args()

    model = load_model(args.model)
    loader = load_data(args.dataset)
    if args.model in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_post = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    model.to(device)
    foveation = foveation.to(device)

    acc = evaluate(model, loader, foveation, T_post, device)

    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()