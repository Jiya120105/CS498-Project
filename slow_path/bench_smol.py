#!/usr/bin/env python3
import argparse, time, os, sys
from PIL import Image
import numpy as np

# Make the package import work whether run as a script or module
if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
from slow_path.service.models.loader import load_model


def make_images(n, w=224, h=224):
    imgs = []
    for i in range(n):
        arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None, help="'cpu', 'cuda', or 'cuda:idx'")
    ap.add_argument('--batches', type=str, default='1,4,8', help='comma list of batch sizes to test')
    args = ap.parse_args()

    if args.device:
        import os
        os.environ['SLOWPATH_DEVICE'] = args.device

    print("Loading SmolVLM model...")
    model = load_model('smol')

    sizes = [int(x) for x in args.batches.split(',') if x]
    prompt = "Is this a person with a backpack? Answer Yes or No."

    for n in sizes:
        imgs = make_images(n)
        prompts = [prompt] * n
        t0 = time.perf_counter()
        outs = model.batch_infer(imgs, prompts)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"batch={n}: total={dt:.1f} ms, per_sample={dt/n:.1f} ms; first={outs[0] if outs else None}")


if __name__ == '__main__':
    main()
