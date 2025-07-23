import torch
import jittor as jt

YOUR_CLIP_PATH = '/defaultShare/archive/wangenguang/.cache/clip'
clip = torch.load(YOUR_CLIP_PATH).state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, './jittor-ViT-B-16.pkl')