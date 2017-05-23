import numpy as np
import argparse
import json
from PIL import Image

def img_l2_loss(lhs_dir, rhs_dir):

    lhs = np.array(Image.open(lhs_dir), dtype=np.float32)
    rhs = np.array(Image.open(rhs_dir), dtype=np.float32)

    return np.linalg.norm(lhs - rhs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--img1', required=True, help="img1 dir")
    parser.add_argument('--img2', required=True, help="img2 dir")

    args = parser.parse_args()
    params = vars(args)
    print json.dumps(params, indent=2)

    print ("L2 Norm between %s and %s is %lf") % (params["img1"], params["img2"], img_l2_loss(params["img1"], params["img2"]))
