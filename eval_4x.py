from __future__ import print_function
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import math
from basicsr.archs.mpenet_arch import MPENet
import numpy as np
import socket
import time
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example - Multi-frame')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='test_data/LR',
                    help='data directory containing sequence folders')
parser.add_argument('--model_type', type=str, default='MPENet')
parser.add_argument('--pretrained_sr', default='experiments/pretrained_models/net_g_145000.pth',
                    help='sr pretrained base model')
parser.add_argument('--save_folder', default='test_data/result/', help='Location to save checkpoint models')
parser.add_argument('--num_frame', type=int, default=7, help='Number of input frames (must match training)')
parser.add_argument('--tile_size', type=int, default=256, help='Tile size for tiled inference')
parser.add_argument('--overlap', type=int, default=24, help='Overlap for tiled inference')


def tiled_inference_multi_frame(model, lq_sequence, scale=4, tile_size=256, overlap=24):
    """
    多帧输入的滑动窗口推理
    lq_sequence: (t, c, h, w)
    模型输入: (1, t, c, tile_size, tile_size)
    模型输出: (1, t, c, hr_h, hr_w) - 输出多帧
    """
    device = lq_sequence.device
    t, c, h, w = lq_sequence.shape

    # 如果图像尺寸小于等于tile_size，直接用整图推理，不分块
    if h <= tile_size and w <= tile_size:
        print(f"  Image size ({h}x{w}) <= tile_size ({tile_size}), using full image inference")
        # 扩展batch维度: (1, t, c, h, w)
        lq_sequence_batch = lq_sequence.unsqueeze(0)
        with torch.no_grad():
            output = model(lq_sequence_batch)  # (1, t, c, hr_h, hr_w)
        output = output.squeeze(0).cpu()  # (t, c, hr_h, hr_w)
        return output

    # 否则使用滑动窗口
    stride = tile_size - overlap

    out_h, out_w = h * scale, w * scale
    # 输出也是多帧: (t, c, out_h, out_w)
    output = torch.zeros((t, c, out_h, out_w), dtype=torch.float32, device='cpu')
    weight_sum = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device='cpu')

    total_tiles_y = (h + stride - 1) // stride
    total_tiles_x = (w + stride - 1) // stride
    total_tiles = total_tiles_y * total_tiles_x

    pbar = tqdm(total=total_tiles, desc="  分块推理", unit="块", leave=False)

    for y in range(0, h, stride):
        y_end = min(y + tile_size, h)
        for x in range(0, w, stride):
            x_end = min(x + tile_size, w)

            # 切出多帧小方块: (t, c, tile_h, tile_w)
            tile = lq_sequence[:, :, y:y_end, x:x_end]
            tile_h, tile_w = tile.shape[2], tile.shape[3]

            # 填充到固定大小
            pad_h = tile_size - tile_h
            pad_w = tile_size - tile_w
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')

            # 扩展batch维度: (1, t, c, tile_size, tile_size)
            tile = tile.unsqueeze(0)

            with torch.no_grad():
                out_tile = model(tile)  # 输出: (1, t, c, hr_tile_size, hr_tile_size)

            # 裁掉填充部分 (所有帧同时裁)
            hr_tile_h = tile_h * scale
            hr_tile_w = tile_w * scale
            # out_tile: (1, t, c, hr_tile_h, hr_tile_w)
            out_tile = out_tile[:, :, :, :hr_tile_h, :hr_tile_w].cpu()
            # 去掉batch维度: (t, c, hr_tile_h, hr_tile_w)
            out_tile = out_tile.squeeze(0)

            # 拼接权重 (所有帧共享同一个权重)
            h_win = np.hanning(hr_tile_h)
            w_win = np.hanning(hr_tile_w)
            weight_2d = np.outer(h_win, w_win)
            # 创建权重tensor: (1, hr_tile_h, hr_tile_w)
            weight_tile = torch.from_numpy(weight_2d).float().unsqueeze(0)

            # 累积到输出 (对所有帧分别累积)
            for frame_idx in range(t):
                # output[frame_idx, :, y*scale:y_end*scale, x*scale:x_end*scale] 形状: (c, hr_tile_h, hr_tile_w)
                # out_tile[frame_idx] 形状: (c, hr_tile_h, hr_tile_w)
                # weight_tile 形状: (1, hr_tile_h, hr_tile_w)
                # 广播乘法: (c, hr_tile_h, hr_tile_w) * (1, hr_tile_h, hr_tile_w) = (c, hr_tile_h, hr_tile_w)
                output[frame_idx, :, y * scale:y_end * scale, x * scale:x_end * scale] += out_tile[
                                                                                              frame_idx] * weight_tile

            # weight_sum 累积: (1, 1, hr_tile_h, hr_tile_w)
            weight_sum[:, :, y * scale:y_end * scale, x * scale:x_end * scale] += weight_tile.unsqueeze(0)

            pbar.update(1)

    pbar.close()

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    # 对所有帧分别归一化
    for frame_idx in range(t):
        output[frame_idx] = output[frame_idx] / weight_sum
    return output


def load_sequence(seq_path, num_frame, transform):
    """加载一个序列的所有帧"""
    frame_files = sorted([p for p in seq_path.glob("*.png") if p.is_file()])

    if len(frame_files) < num_frame:
        print(f"Warning: {seq_path} has only {len(frame_files)} frames, required {num_frame}")
        # 循环填充
        while len(frame_files) < num_frame:
            frame_files.append(frame_files[-1])

    # 取前num_frame帧
    frame_files = frame_files[:num_frame]

    frames = []
    for img_path in frame_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        frames.append(img_tensor)

    sequence = torch.stack(frames, dim=0)  # (t, c, h, w)
    return sequence


opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True

print(opt)

current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transforms.Compose([transforms.ToTensor()])


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))


torch.cuda.manual_seed(opt.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

print('===> Building model ', opt.model_type)
model = MPENet()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
print('---------- Networks architecture -------------')
print_network(model)
model = model.to(device)

model_name = os.path.join(opt.pretrained_sr)
if os.path.exists(model_name):
    # 加载检查点
    checkpoint = torch.load(model_name, map_location=device)

    # 检查是否是完整的checkpoint（包含'params'和'params_ema'）
    if 'params' in checkpoint:
        # 这是BasicSR训练时保存的完整checkpoint，只取模型权重
        state_dict = checkpoint['params']
        print('Loaded checkpoint with params (using model weights)')
    else:
        # 直接是模型权重
        state_dict = checkpoint
        print('Loaded direct model weights')

    # 处理多GPU保存的模型
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
        print('Removed "module." prefix from state_dict keys')

    # 加载权重
    model.module.load_state_dict(state_dict)
    print('Pre-trained SR model is loaded successfully!')
else:
    print(f'No pre-trained model found at {model_name}!')
    exit()


def eval():
    print('===> Loading test datasets')
    data_dir = Path(opt.data_dir)

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist!")
        return

    # 获取所有序列文件夹
    sequence_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    if len(sequence_folders) == 0:
        print(f"No sequence folders found in {data_dir}")
        print("Expected structure: data_dir/sequence_001/0000.png, 0001.png, ...")
        return

    print(f"Found {len(sequence_folders)} sequences")

    model.eval()

    for seq_idx, seq_path in enumerate(sequence_folders):
        seq_name = seq_path.name
        print(f"\nProcessing sequence [{seq_idx + 1}/{len(sequence_folders)}]: {seq_name}")

        # 加载序列
        sequence = load_sequence(seq_path, opt.num_frame, transform)
        print(f"  Input shape: {sequence.shape}")

        # 移到GPU
        lq_sequence = sequence.to(device)  # (t, c, h, w)

        # 推理
        with torch.no_grad():
            t0 = time.time()
            prediction = tiled_inference_multi_frame(
                model,
                lq_sequence,
                scale=opt.upscale_factor,
                tile_size=opt.tile_size,
                overlap=opt.overlap
            )
            t1 = time.time()

        print(f"  Inference time: {t1 - t0:.4f} sec")
        print(f"  Output shape: {prediction.shape}")  # (t, c, hr_h, hr_w)

        # 保存所有帧
        save_folder = opt.save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 为当前序列创建子文件夹
        seq_save_folder = os.path.join(save_folder, seq_name)
        if not os.path.exists(seq_save_folder):
            os.makedirs(seq_save_folder)

        # 遍历所有帧
        for frame_idx in range(prediction.shape[0]):
            # 提取单帧: (c, hr_h, hr_w)
            frame = prediction[frame_idx:frame_idx + 1]  # (1, c, hr_h, hr_w)

            # 后处理
            frame = frame.cpu()
            frame = frame.data[0].numpy().astype(np.float32)  # (c, hr_h, hr_w)
            frame = frame * 255.0
            frame = frame.clip(0, 255)
            frame = frame.transpose(1, 2, 0)  # (hr_h, hr_w, c)

            # 保存每一帧，命名方式：序列名_帧索引.png
            save_fn = os.path.join(seq_save_folder, f'{seq_name}_frame{frame_idx:04d}_SR.png')
            Image.fromarray(np.uint8(frame)).save(save_fn)

        print(f"  Saved all {prediction.shape[0]} frames for sequence {seq_name}")


if __name__ == '__main__':
    eval()