import numpy as np


# 通过滑窗来取得小的数据样本
#input_data的形状为(C,H,W)
def split_data(input_data, win_h=64, win_w=64, win_c=64, win_shift=32):
    C, H, W = input_data.shape
    blocks = np.zeros((C//win_shift-1, H//win_shift-1, W//win_shift-1, win_c, win_h, win_w))
    for c in range(C//win_shift-1):
        for h in range(H//win_shift-1):
            for w in range(W//win_shift-1):
                blocks[c, h, w] = input_data[win_shift*c:win_shift*c+win_c,
                                             win_shift*h:win_shift*h+win_h,
                                             win_shift*w:win_shift*w+win_w]
    return blocks


# Concat 将上面得到的blocks类型的数据重新拼合起来
# data的形状为 (c,h,w,win_c,win_h,win_w),c,h,w分别表示每通道，每行，每列各有多少个分出来的块，块的形状为(win_c,win_h,win_w)
# ori_shape 为源图像的形状，即 (333,512,512)之类
def Concat(data, ori_shape, win_shift):
    C, H, W = data.shape[:3]
    output = np.zeros(ori_shape)
    for c in range(C):
        H_tmp = np.zeros((data.shape[3], ori_shape[1], ori_shape[2]))
        for h in range(H):
            W_tmp = np.zeros((data.shape[3], data.shape[4], ori_shape[2]))
            W_tmp[:, :, :data.shape[5]] = data[c, h, 0]
            for w in range(1, W):
                W_tmp[:, :, w*win_shift:(w-1)*win_shift+data.shape[5]] += data[c, h, w, :, :, :data.shape[5]-win_shift]
                W_tmp[:, :, w*win_shift:(w-1)*win_shift+data.shape[5]] /= 2
                W_tmp[:, :, (w-1)*win_shift+data.shape[5]:(w-1)*win_shift+2*data.shape[5]-win_shift]\
                    = data[c, h, w, :, :, data.shape[5]-win_shift:]
            if h == 0:
                H_tmp[:, :data.shape[4], :] = W_tmp
                continue
            H_tmp[:, h*win_shift:(h-1)*win_shift+data.shape[4], :] += W_tmp[:, :data.shape[4]-win_shift, :]
            H_tmp[:, h*win_shift:(h-1)*win_shift+data.shape[4], :] /= 2
            H_tmp[:, (h-1)*win_shift+data.shape[4]:(h-1)*win_shift+2*data.shape[4]-win_shift,:]\
                = W_tmp[:, data.shape[4]-win_shift:, :]
        if c == 0:
            output[:data.shape[3], :, :] = H_tmp
            continue
        output[c*win_shift:(c-1)*win_shift+data.shape[3]] += H_tmp[:data.shape[3]-win_shift, :, :]
        output[c*win_shift:(c-1)*win_shift+data.shape[3]] /= 2
        output[(c-1)*win_shift+data.shape[3]:(c-1)*win_shift+2*data.shape[3]-win_shift, :, :]\
            = H_tmp[data.shape[3]-win_shift:, :, :]
    return output
