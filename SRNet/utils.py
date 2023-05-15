import os, cv2
import numpy as np
from PIL import Image, ImageDraw

def calc_iou(p, t):
    mul = (p*t).sum()
    add = (p+t).sum()
    return mul / (add - mul + 1e-6)


def debugDump(output_dir, image_name, texts, lsts, size=(256, 256)):
    """
    Args:
        texts: list of list of text
        lsts: list of list of torch.Tensor H, W
    """
    os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)
    outs = []
    for txts, lst in zip(texts, lsts):
        lst = [cv2.resize((x.numpy()*255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR) for x in lst]
        lst = [Image.fromarray(x) for x in lst]
        for x, t in zip(lst, txts):
            ImageDraw.Draw(x).text((0, 0), str(t), fill="red")
        out = Image.fromarray(np.concatenate([np.array(x) for x in lst], axis=1))
        outs.append(np.array(out))
    out = Image.fromarray(np.concatenate(outs, axis=0))
    out.save(os.path.join(output_dir, "debug", image_name+".png"))