import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
from torchvision import transforms
from torch.nn import functional as F

img_path = f'../DNA/DNA/dataset/track1/images/03534.png'

def window_partition(x, win_size, dilation_rate=1):
    B, C, H, W = x.shape
    if dilation_rate != 1:
        # x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.contiguous()  # B',C ,Wh ,Ww
    else:
        x = x.permute(0, 2, 3, 1).view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(-1, C, win_size, win_size)  # B',C ,Wh ,Ww
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B',C ,Wh ,Ww
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    B = 1 if B == 0 else B
    x = windows.view(B,  H // win_size, W // win_size, -1, win_size, win_size)
    if dilation_rate != 1:
        x = windows.permute(0, 3, 4, 5, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x

img = Image.open(img_path).convert('RGB')

width, height = img.size
h, w = img.size
        # 计算需要扩展的宽度和高度
new_width = ((width // 512) + 1) * 512
new_height = ((height // 512) + 1) * 512

# 计算需要扩展的像素数
pad_width = new_width - width
pad_height = new_height - height

# 使用ImageOps.expand扩展图片
img = ImageOps.expand(img, border=(0, 0, pad_width, pad_height), fill=0)
print(img.size)
H, W = img.size

input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.26091782370802136, 0.26091782370802136, 0.26091782370802136], [0.017222260313435774
, 0.017222260313435774, 0.017222260313435774])
    ])


img = input_transform(img)
img = window_partition(img.unsqueeze(0), 512)

img = window_reverse(img, 512, W, H)
print(img.shape)
img = img[:, :, :w, :h]
img = img.permute(0, 2, 3, 1).numpy()
img = img[0, :, :, :]
img = img / 2 + 0.5
img = (img * 255.).astype(np.uint8)
cv2.imshow('img', img)
cv2.waitKey(0)


