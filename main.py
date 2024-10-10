import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import models
import pywt
import time

def batched_predict(model, inp, inp_dct, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp, inp_dct)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def time_text(t):
    if t >= 3600:
        return '{:.3f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.3f}m'.format(t / 60)
    else:
        return '{:.3f}s'.format(t)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def get_dwt(img):
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    crop_hr_Y_dwt = np.array([[cA, cH, cV, cD]], dtype='float32').transpose([0, 2, 3, 1])
    crop_hr_Y_dwt_tensor = transforms.ToTensor()(crop_hr_Y_dwt[0])
    return crop_hr_Y_dwt_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='0904x2.png')
    parser.add_argument('--model', default='pretrain_model/iwt.pth')
    parser.add_argument('--scale', default='2')
    parser.add_argument('--output', default='output0904x2.png')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
 
    # 自动检测是否有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    scale_max = 4
    img_pillow = Image.open(args.input)
    img = transforms.ToTensor()(img_pillow)
    img_np = np.array(img_pillow)
    img_dwt_tensor = get_dwt(img_np / 255)

    # 加载模型并将其移动到设备
    model = models.make(torch.load(args.model, map_location=torch.device(device))['model'], load_sd=True).to(device)

    t1 = time.time()
    h = int(img.shape[-2] * int(args.scale))
    w = int(img.shape[-1] * int(args.scale))
    scale = h / img.shape[-2]
    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    cell_factor = max(scale / scale_max, 1)

    # 预测时将所有相关张量移动到设备
    pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
                        img_dwt_tensor.to(device).unsqueeze(0), coord.unsqueeze(0),
                        (cell_factor * cell).unsqueeze(0), bsize=30000).squeeze(0)

    pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(1, h, w).cpu()
    transforms.ToPILImage()(pred).save(args.output)

    t2 = time.time()
    print(time_text(t2 - t1))
