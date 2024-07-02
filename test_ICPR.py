try:
    from collections import OrderedDict
    
    import numpy as np
    from PIL import Image, ImageOps, ImageFilter
    import cv2
    from torchvision import transforms
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from argparse import ArgumentParser
    import einops
    import torch
    import torch.nn as nn
    # from model.utils import get_2d_sincos_pos_embed
    
    import math
    from einops import rearrange, repeat
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    import torch.nn.functional as F
    import os
except ImportError as e:
    print("模块导入错误：", e)
except ModuleNotFoundError as e:
    print("模块未找到错误：", e)
except Exception as e:
    print("导入过程中发生了其他错误：", e)
    
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--model_dir', type=str, default='./Best_nIoU_Epoch-160_IoU-0.0228_nIoU-0.1313.pth.tar')
    parser.add_argument('--save_dir', type=str, default='./mask')
    parser.add_argument('--bs', type=int, default=4)
    #
    # Training parameters
    #
    # Net parameters
    #
    #
    # Dataset parameters
    #
    args = parser.parse_args()
    return args


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,
                 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                     [0.26091782370802136, 0.26091782370802136, 0.26091782370802136],
                     [0.017222260313435774, 0.017222260313435774, 0.017222260313435774])]),
                 base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.images    = dataset_dir+'/'+'images'
        self.images2 = dataset_dir + '/' + 'img'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img):
        # base_size = self.base_size
        # img  = img.resize ((base_size, base_size), Image.BICUBIC)
        # mask = mask.resize((base_size, base_size), Image.NEAREST)
        width, height = img.size

        # 计算需要扩展的宽度和高度
        new_width = ((width // 512) + 1) * 512
        new_height = ((height // 512) + 1) * 512

        # 计算需要扩展的像素数
        pad_width = new_width - width
        pad_height = new_height - height

        # 使用ImageOps.expand扩展图片
        img = ImageOps.expand(img, border=(0, 0, pad_width, pad_height), fill=0)



        # final transform
          # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img

    def __getitem__(self, idx):
        # print('idx:',idx)
        try:
            idx = idx.split('.')[0]
        except:
            idx = idx
        cv2.setNumThreads(0)

        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        try:
            img_id = img_id.split('.')[0]
        except:
            img_id = img_id
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        img_path2  = self.images2+'/'+img_id+self.suffix
        try:
            img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸

        except FileNotFoundError as e:
    # 文件未找到错误
            print("文件未找到：", e)
        except Exception as e:
    # 其他所有类型的异常
            print("加载过程中发生了错误：", e)
        h, w = img.size
        # synchronized transform
        img= self._testval_sync_transform(img)
        H, W = img.size
        img= np.array(img)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        # print('shape1', mask.shape)
        img = window_partition_test(img.unsqueeze(0), self.crop_size)


        d = np.array((H, W, h, w, self.crop_size)).astype('float32')
        d = torch.from_numpy(d)

        return img, d  # img_id[-1]

    def __len__(self):
        return len(self._items)

def window_partition_test(x, win_size, dilation_rate=1):
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



def window_reverse_test(windows, win_size, H, W, dilation_rate=1):
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



def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    B = 1 if B == 0 else B
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def bchw_to_blc(x):
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x


def blc_to_bchw(x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).view(B, C, H, W)
    return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        padding = (kernel_size - stride) // 2
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         # dilation=dilation,
                                         groups=in_channels
                                         )
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


class ConvProjection(nn.Module):
    def __init__(self, dim, co_dim, heads=8, dim_head=64, kernel_size=3, q_stride=1,
                 bias=True, hidden_dim=None):
        super().__init__()
        self.res = None
        inner_dim = dim_head * heads if hidden_dim is None else hidden_dim
        self.heads = heads
        pad = (kernel_size - q_stride) // 2

        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)

        self.to_k = SepConv2d(co_dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_v = SepConv2d(co_dim, inner_dim, kernel_size, q_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, _, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, mode, co_dim, heads=8, dim_head=64, dropout=0., bias=True, hidden_dim=None, win_size=8):
        super().__init__()
        inner_dim = dim_head * heads if hidden_dim is None else hidden_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias) if mode != 'cross' else nn.Linear(dim // 2, inner_dim, bias=bias)
        self.to_kv = nn.Linear(co_dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim
        self.mode = mode
        self.cond = nn.Parameter(torch.randn(win_size[0], win_size[1] // 2 + 1, dim, 2, dtype=(torch.float32)) * 0.02)
        # nn.init.trunc_normal_(self.condition, std=0.02)


    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None or self.mode == 'task':

            if self.mode == 'task':
                # attn_kv = x * attn_kv
                x_ = blc_to_bchw(x).permute(0, 2, 3, 1)
                B, H, W, C = x_.shape
                X = torch.fft.rfft2(x_, dim=(1, 2), norm='ortho')
                cond = torch.view_as_complex(self.cond)
                X = X * cond
                attn_kv = torch.fft.irfft2(X, s=(H, W), dim=(1, 2), norm='ortho')
                attn_kv = attn_kv.permute(0, 3, 1, 2)
                attn_kv = bchw_to_blc(attn_kv)
        else:

            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(attn_kv).reshape(B_, N, 1, self.heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(x).reshape(B_, N_kv, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        q = q[0]
        v, k = kv[0], kv[1]

        return q, k, v


##          FF          ###
#
#
#

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        if in_planes <= ratio:
            ratio = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reduction = reduction

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, outdim=32, act_layer=nn.GELU, drop=0., use_eca=True ,downsample=1):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim//downsample, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        # self.pwconv = nn.Sequential(nn.Conv2d(hidden_dim, outdim, 1))
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, outdim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        # self.eca = eca_layer_1d(outdim) if use_eca else nn.Identity()
        # self.ca = CBAM(outdim)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        #
        x = self.dwconv(x)
        # x = self.pwconv(x)
        #
        # x = self.ca(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        # x = self.eca(x)

        return x


class GatedFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_se=True ,downsample=1):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim//downsample, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.outconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x1 = self.dwconv(x)
        x2 = self.pwconv(x)
        x = x1 * x2
        x = self.outconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)

        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., downsample=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features // downsample, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class WinCoTransLayer(nn.Module):
    def __init__(self,
                 dim,
                 sigma_dim,
                 input_resolution,
                 win_size=8,
                 shift_size=0,
                 heads=1,
                 depth=2,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path=[2],
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',
                 token_mlp='leff',
                 mode=None
                 ):
        super(WinCoTransLayer, self).__init__()
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.shift_size = shift_size
        self.pool = nn.Identity()
        self.pool_co = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.input_proj = nn.Sequential(nn.Conv2d(sigma_dim, dim, 3, 1, 1),
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(inplace=True),
                                        )

        self.depth = depth

        self.layer = nn.ModuleList([])
        self.layer2 = nn.ModuleList([])
        self.conv = nn.ModuleList([])
        self.patch_embed = nn.ModuleList([])
        self.di0 = nn.ModuleList([])
        self.di2 = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.ca = nn.ModuleList([])
        self.patch_proj = nn.ModuleList([])
        self.dwconv = nn.ModuleList([])
        self.pwconv = nn.ModuleList([])
        self.mode = mode
        modulator = nn.ParameterList([])
        for i in range(depth):
            num_heads = heads[i] if isinstance(heads, list) else heads
            dp = drop_path[i] if isinstance(drop_path, list) else drop_path


            self.layer.append(
                WindowCoTransEncoder(dim=dim, input_resolution=(input_resolution, input_resolution), num_heads=num_heads,
                                     win_size=win_size,
                                     shift_size=max(0, self.shift_size) if (i % 2 == 0) else win_size//2,

                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_ratio,
                                     attn_drop=attn_drop_ratio, drop_path=dp, act_layer=act_layer, norm_layer=norm_layer,
                                     token_projection=token_projection, token_mlp=token_mlp, downsample=1,mode=mode)
            )
            self.di0.append(nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)))
            self.di2.append(nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=2, dilation=2), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)))
            self.conv.append(nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)))
            self.ca.append(CBAM(dim))
            self.act.append(nn.ReLU(inplace=True))
            self.dwconv.append(nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, 1, dim), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)))
            self.pwconv.append(nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim)))



    def init(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], int(self.num_patches ** .5),
                                            cls_token=False)
        self.pos_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        pass

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = 1
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def forward(self, x, attn_kv=None):
        x = self.input_proj(x)
        H = W = x.shape[2]

        # TODO: Pos_emb
        # DENIED


        for i in range(self.depth):


            res = x
            di0 = self.di0[i](x)
            di2 = self.di2[i](x)

            x = self.conv[i](torch.cat([di0, di2], dim=1))
            shortcut = x
            x = bchw_to_blc(x)
            x = self.pool(x)
            x = self.layer[i](x, attn_kv=attn_kv)
            x = blc_to_bchw(x)
            shortcut = self.dwconv[i](shortcut)
            x = shortcut * x
            x = self.pwconv[i](x)
            x = self.ca[i](x)
            x = x + res
            x = self.act[i](x)


        return x


###     Window attn     ###
#
#
#
class CoWindowAttention(nn.Module):
    def __init__(self, dim, co_dim, win_size, num_heads, mode, token_projection='conv', qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., dowmsample=4):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.pos_emb = nn.Parameter(torch.zeros(1, 1, self.win_size[0] ** 2, self.win_size[0] ** 2))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim=dim, co_dim=co_dim, heads=num_heads,
                                      dim_head=dim // num_heads // dowmsample, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim=dim, co_dim=co_dim, heads=num_heads, mode=mode, win_size=win_size,
                                        dim_head=dim // num_heads // dowmsample, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // dowmsample, dim // dowmsample)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape

        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + self.pos_emb
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=1)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * 1) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * 1)
            attn = self.softmax(attn)

        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowCoTransEncoder(nn.Module):
    def __init__(self, dim, mode, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='conv', token_mlp='leff', downsample=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim // downsample)
        self.norm2 = norm_layer(dim // downsample // 2) if mode == 'cross' else nn.Identity()
        self.norm3 = norm_layer(dim // downsample)
        self.attn = CoWindowAttention(dim=dim // downsample, co_dim=dim // downsample, win_size=to_2tuple(self.win_size),
                                      num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mode=mode,
                                      attn_drop=attn_drop, proj_drop=drop, token_projection=token_projection, dowmsample=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        if token_mlp in ['ffn', 'mlp', 'Mlp']:
            self.mlp = Mlp(in_features=dim // downsample, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, downsample=1)
        elif token_mlp in ['leff', 'Leff']:
            self.mlp = LeFF(dim // downsample, mlp_hidden_dim, outdim=dim, act_layer=act_layer, drop=drop, downsample=1)
        elif token_mlp in ['gate', 'Gate']:
            self.mlp = GatedFF(dim // downsample, mlp_hidden_dim, act_layer=act_layer, drop=drop, downsample=1)
        elif token_mlp in ['fastleff', 'Fastleff']:
            # self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            pass
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, attn_kv=None, mask=None):

        B, L, C = x.shape
        if attn_kv is not None:
            _, _, C_co = attn_kv.shape

        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        attn_mask = mask

        ## shift mask

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = shift_attn_mask

        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if attn_kv is not None:
            attn_kv = self.norm2(attn_kv)
            attn_kv = attn_kv.view(B, H, W, C_co)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if attn_kv is not None:
                shifted_co = torch.roll(attn_kv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        else:
            shifted_x = x
            if attn_kv is not None:
                shifted_co = attn_kv

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        co_windows = None
        if attn_kv is not None:
            co_windows = window_partition(shifted_co, self.win_size)
            co_windows = co_windows.view(-1, self.win_size * self.win_size, C_co)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, co_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        else:
            x = shifted_x
        x = x.view(B, H * W, -1)

        # FFN
        x = self.drop_path(x)
        xx = torch.sigmoid(self.mlp(self.norm3(x)))
        x = xx
        del attn_mask

        return x


###########
#
#
###########


class WindowPBTNet(nn.Module):
    def __init__(self, num_classes, input_channels, num_blocks, nb_filter, deep_supervision,
                 depth, heads, win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 token_projection='linear', token_mlp='leff', img_size=256
                 ):
        super(WindowPBTNet, self).__init__()

        heads = heads or [[2, 2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2], [2, 2], [2]]
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)


        self.down1 = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=4, stride=2, padding=1)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        dpr4 = [x.item() for x in torch.linspace(0, drop_path, depth[4])]
        dpr3 = [x.item() for x in torch.linspace(0, drop_path, depth[3] * 2)]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path, depth[2] * 3)]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path, depth[1] * 4)]
        dpr0 = [x.item() for x in torch.linspace(0, drop_path, depth[0] * 5)]


         # 11222
        self.conv0_0 = WinCoTransLayer(sigma_dim=input_channels, dim=nb_filter[0], depth=2, heads=heads[0][0], mode='task',
                                       drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=0.,
                                       token_projection=token_projection, token_mlp=token_mlp,
                                       win_size=win_size, input_resolution=img_size)
        self.conv1_0 = WinCoTransLayer(sigma_dim=nb_filter[1], dim=nb_filter[1], depth=2, heads=heads[1][0], mode='task',
                                       drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=0.,
                                       token_projection=token_projection, token_mlp=token_mlp,
                                       win_size=win_size, input_resolution=img_size // 2)
        self.conv2_0 = WinCoTransLayer(sigma_dim=nb_filter[2], dim=nb_filter[2], depth=2, heads=heads[2][0], mode='task',
                                       drop_ratio=drop, attn_drop_ratio=attn_drop, drop_path=0.,
                                       token_projection=token_projection, token_mlp=token_mlp,
                                       win_size=win_size * 2, input_resolution=img_size // 4)
        self.conv3_0 = WinCoTransLayer(sigma_dim=nb_filter[3], dim=nb_filter[3], depth=2, mode='task',
                                       drop_ratio=drop, attn_drop_ratio=attn_drop, heads=heads[3][0], drop_path=0.,
                                       act_layer=act_layer, token_projection=token_projection, token_mlp=token_mlp,
                                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size * 4, input_resolution=img_size // 8)
        self.conv4_0 = WinCoTransLayer(sigma_dim=nb_filter[4], dim=nb_filter[4], depth=4, mode='task',
                                       drop_ratio=drop, attn_drop_ratio=attn_drop, heads=heads[4][0], drop_path=0.,
                                       act_layer=act_layer, token_projection=token_projection, token_mlp=token_mlp,
                                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size * 4, input_resolution=img_size // 16)

        self.attn0_1 = WinCoTransLayer(sigma_dim=nb_filter[0] + nb_filter[1], dim=nb_filter[0], depth=depth[0],
                                       heads=heads[0][1], drop_ratio=drop, attn_drop_ratio=attn_drop,mode='cross',
                                       drop_path=dpr0[0:depth[0]], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size//2, input_resolution=img_size, shift_size=win_size//4)

        self.attn1_1 = WinCoTransLayer(sigma_dim=nb_filter[0] + nb_filter[1] + nb_filter[2], dim=nb_filter[1],mode='cross',
                                       depth=depth[1], heads=heads[1][1], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr1[0:depth[1]], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size, input_resolution=img_size // 2, )

        self.attn2_1 = WinCoTransLayer(sigma_dim=nb_filter[1] + nb_filter[2] + nb_filter[3], dim=nb_filter[2],mode='cross',
                                       depth=depth[2], heads=heads[2][1], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr2[0:depth[2]], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size * 2, input_resolution=img_size // 4)

        self.attn3_1 = WinCoTransLayer(sigma_dim=nb_filter[2] + nb_filter[3] + nb_filter[4],
                                       dim=nb_filter[3], depth=depth[3], drop_ratio=drop,
                                       attn_drop_ratio=attn_drop, heads=heads[3][1], drop_path=dpr3[0:depth[3]],
                                       act_layer=act_layer, token_projection=token_projection, token_mlp=token_mlp,
                                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size * 4, input_resolution=img_size // 8)

        self.attn0_2 = WinCoTransLayer(sigma_dim=nb_filter[0] * 2 + nb_filter[1], dim=nb_filter[0],mode='cross',
                                       depth=depth[0], heads=heads[0][2], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr0[depth[0]:depth[0] * 2], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size//2, input_resolution=img_size)

        self.attn1_2 = WinCoTransLayer(sigma_dim=nb_filter[0] + nb_filter[1] * 2 + nb_filter[2], dim=nb_filter[1],mode='cross',
                                       depth=depth[1], heads=heads[1][2], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr1[depth[1]:depth[1] * 2], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size, input_resolution=img_size // 2)

        self.attn2_2 = WinCoTransLayer(sigma_dim=nb_filter[1] + nb_filter[2] * 2 + nb_filter[3], dim=nb_filter[2],
                                       depth=depth[2], heads=heads[2][2], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr2[depth[2]:depth[2] * 2], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size * 2, input_resolution=img_size // 4)

        self.attn0_3 = WinCoTransLayer(sigma_dim=nb_filter[0] * 3 + nb_filter[1], dim=nb_filter[0], depth=depth[0],mode='cross',
                                       heads=heads[0][3], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr0[depth[0] * 2:depth[0] * 3], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size//2 if img_size == 256 else win_size, input_resolution=img_size,
                                       shift_size=win_size//4 if img_size == 256 else win_size//2)

        self.attn1_3 = WinCoTransLayer(sigma_dim=nb_filter[0] + nb_filter[1] * 3 + nb_filter[2], dim=nb_filter[1],
                                       depth=depth[1], heads=heads[1][3], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=dpr1[depth[1] * 2:depth[1] * 3], act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size, input_resolution=img_size // 2, )

        self.attn0_4 = WinCoTransLayer(sigma_dim=nb_filter[0] * 4 + nb_filter[1], dim=nb_filter[0],
                                       depth=depth[0], heads=heads[0][4], drop_ratio=drop, attn_drop_ratio=attn_drop,
                                       drop_path=drop_path, act_layer=act_layer,
                                       token_projection=token_projection,
                                       token_mlp=token_mlp, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       win_size=win_size//2 if img_size == 256 else win_size, input_resolution=img_size)

        self.shuffle = nn.PixelShuffle(2)
        self.pool_co = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def attn_kv_transform(self, attn_kv):

        attn_kv = self.shuffle(attn_kv)
        attn_kv = bchw_to_blc(attn_kv)

        return attn_kv

    def forward(self, input,return_attn = None):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.down1(x0_0))
        x2_0 = self.conv2_0(self.down2(x1_0))
        x3_0 = self.conv3_0(self.down3(x2_0))
        x4_0 = self.conv4_0(self.down4(x3_0), attn_kv=None)

        x3_1 = self.attn3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_0)], 1), attn_kv=None)# x4_1
        x2_1 = self.attn2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_0)], 1), attn_kv=self.attn_kv_transform(x3_1))# x3_1
        x1_1 = self.attn1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_0)], 1), attn_kv=self.attn_kv_transform(x2_1))# x2_1
        x0_1 = self.attn0_1(torch.cat([x0_0, self.up(x1_0)], 1), attn_kv=self.attn_kv_transform(x1_1))#x1_1

        x2_2 = self.attn2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_1)], 1), attn_kv=None)# x3_2
        x1_2 = self.attn1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_1)], 1), attn_kv=self.attn_kv_transform(x2_2))# x2_2
        x0_2 = self.attn0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1), attn_kv=self.attn_kv_transform(x1_2))

        x1_3 = self.attn1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_2)], 1), attn_kv=None)# x2_3
        x0_3 = self.attn0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1), attn_kv=self.attn_kv_transform(x1_3))# x1_3

        x0_4 = self.attn0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1), attn_kv=None)# x1_4




        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output5 = self.final5(x0_4)

            return [output1, output2, output3, output5]
        else:
            output = self.final(x0_4)

            return output


def main(args):
    dataset_root = args.dataset
    test_txt = dataset_root + '/' + 'img_idx' + '/' + 'test.txt'
    model_dir = args.model_dir
    save_dir = args.save_dir
    bs = args.bs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_ids = []
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    valset = TestSetLoader(dataset_root, img_id=img_ids , base_size=512, crop_size=512)
    val_data = DataLoader(dataset=valset, batch_size=1, num_workers=0, drop_last=False, shuffle=False)
    model = WindowPBTNet(num_classes=1, input_channels=3, num_blocks=[2, 2, 2, 2, 2],
                         nb_filter=[16, 32, 64, 128, 256, 512, 1024],
                         deep_supervision=False, depth=[2, 2, 2, 2, 2], drop=0.1, attn_drop=0.1, drop_path=0.1,
                         mlp_ratio=4.,
                         # heads=[[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1], [1]], token_projection='linear',
                         heads=[[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1], [1]], token_projection='linear',
                         token_mlp='leff', win_size=8, img_size=512, )
    model = model.to(device)
    print('model create')
    try:
        ckpt = torch.load(model_dir, map_location=device)
    except FileNotFoundError as e:
    # 文件未找到错误
        print("文件未找到：", e)
    except Exception as e:
    # 其他所有类型的异常
        print("加载过程中发生了错误：", e)
    model_weights = ckpt['state_dict']
    try:
        model.load_state_dict(ckpt['state_dict'], strict=True)
    except:
        new_dict = OrderedDict()
        for k, v in model_weights.items():
            name = k[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict, strict=True)

    print('model loaded')
    with torch.no_grad():
        model.eval()
        for step, data in enumerate(val_data):
            print(step)
            images, d = data
            images, d = images[0, :, :, :, :], d[0]
            H, W, h, w, size = d[0], d[1], d[2], d[3], d[4]
            num_iter = (images.shape[0]) // bs  if images.shape[0] % bs == 0 else (images.shape[0]) // bs + 1
            print(num_iter)
            predlist = []
            for i in range(num_iter):
                img = images[i * bs:(i + 1) * bs, :, :, :] if (i + 1) * bs < images.shape[0] else images[i * bs:images.shape[0], :, :, :]
                pred = model(img.to(device))
                if isinstance(pred, list):
                    pred = pred[-1]
                predlist.append(pred)
            pred = torch.cat(predlist, dim=0)
            pred = window_reverse_test(pred, int(size.item()), int(W.item()), int(H.item()))
            pred = pred[:, :, :int(w.item()), :int(h.item())]
            pred[pred < 0] = 0
            pred[pred > 0] = 1
            pred = pred[0, 0, :, :]
            try:
                pred = pred.detach().cpu().numpy()
            except:
                pred = pred.detach().numpy()
            pred = (pred * 255).astype(np.uint8)

            try:
                id = img_ids[step].split('.')[0]
            except:
                id = img_ids[step]

            cv2.imwrite(save_dir + '/' + id + '.png', pred)
            print(save_dir + '/' + id +'.png')





if __name__ == '__main__':
    args = parse_args()
    main(args)


# python test_ICPR.py --dataset=../DNA/DNA/dataset/track1
# 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
