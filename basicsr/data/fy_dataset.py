import random
import torch
from pathlib import Path
from torch.utils import data as data
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor

from basicsr.data.data_util import read_img_seq
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class SimpleSequenceDataset(data.Dataset):
    """简化的序列数据集，用于训练循环网络。

    数据结构示例：
    train/
    ├── GT/
    │   ├── 00000/
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   ├── ...
    │   │   └── 0007.png
    │   ├── 00001/
    │   └── ...
    └── LR/
        ├── 00000/
        │   ├── 0000.png
        │   ├── 0001.png
        │   ├── ...
        │   └── 0007.png
        ├── 00001/
        └── ...

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (bool): Scale, which will be added automatically.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SimpleSequenceDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        # 生成所有keys（不使用元数据文件，直接扫描文件夹）
        self.keys = self._generate_keys()

        # 文件客户端 (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)

        # 获取logger（如果basicsr的logger不可用，可以注释掉或替换）
        try:
            from basicsr.utils import get_root_logger
            logger = get_root_logger()
            logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                        f'random reverse is {self.random_reverse}.')
        except:
            print(f'Temporal augmentation interval list: [{interval_str}]; '
                  f'random reverse is {self.random_reverse}.')

    def _generate_keys(self):
        """生成所有keys，格式为 'sequence_name/frame_index'"""
        keys = []

        # 获取所有序列文件夹
        sequences = [p for p in self.gt_root.iterdir() if p.is_dir()]
        sequences.sort()  # 排序保证一致性

        for seq_path in sequences:
            seq_name = seq_path.name
            # 获取该序列中的所有帧
            frame_files = sorted(seq_path.glob("*.png"))
            frame_count = len(frame_files)

            # 生成keys
            for i in range(frame_count):
                keys.append(f'{seq_name}/{i:04d}')

        return keys

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: "00000/0000"

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        # 获取当前序列的总帧数
        total_frames = self._get_sequence_length(clip_name)

        if start_frame_idx > total_frames - self.num_frame * interval:
            start_frame_idx = random.randint(0, total_frames - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:04d}'
                img_gt_path = f'{clip_name}/{neighbor:04d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:04d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:04d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, str(img_gt_path))

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

    def _get_sequence_length(self, clip_name):
        """获取指定序列的帧数"""
        if self.is_lmdb:
            # LMDB模式需要特殊处理，这里简化处理
            # 实际使用中可能需要从LMDB读取元信息
            return 8  # 默认返回8，实际应该从数据库获取
        else:
            # 磁盘模式：统计文件夹中的png文件数量
            gt_seq_dir = self.gt_root / clip_name
            return len(list(gt_seq_dir.glob("*.png")))

@DATASET_REGISTRY.register()
class SimpleVideoRecurrentTestDataset(data.Dataset):
    """Video test dataset for recurrent architectures, adapted for your data format.

    Data structure:
        dataroot
        ├── GT
        │   ├── 00000
        │   │   ├── 0000.png
        │   │   ├── 0001.png
        │   │   └── ...
        │   └── 00001
        └── LR
            ├── 00000
            │   ├── 0000.png
            │   ├── 0001.png
            │   └── ...
            └── 00001

    Args:
        opt (dict): Config for test dataset. It contains:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            cache_data (bool): Whether to cache testing datasets.
            io_backend (dict): IO backend type.
    """

    def __init__(self, opt):
        super(SimpleVideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root = Path(opt['dataroot_gt'])
        self.lq_root = Path(opt['dataroot_lq'])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        # Generate data info by scanning folders
        self.imgs_lq = {}
        self.imgs_gt = {}
        self.folders = self._scan_folders()

        if self.cache_data:
            self._cache_all_data()

        logger = get_root_logger()
        logger.info(f'SimpleVideoRecurrentTestDataset initialized with {len(self.folders)} folders')

    def _scan_folders(self):
        """Scan folders and return sorted folder names"""
        folders = [p.name for p in self.lq_root.iterdir() if p.is_dir()]
        folders.sort()
        return folders

    def _cache_all_data(self):
        """Cache all sequences for each folder"""
        logger = get_root_logger()

        for folder in self.folders:
            logger.info(f'Caching {folder}...')

            # Cache LQ sequence
            lq_folder = self.lq_root / folder
            lq_paths = sorted([str(p) for p in lq_folder.glob("*.png")])
            self.imgs_lq[folder] = read_img_seq(lq_paths)

            # Cache GT sequence
            gt_folder = self.gt_root / folder
            gt_paths = sorted([str(p) for p in gt_folder.glob("*.png")])
            self.imgs_gt[folder] = read_img_seq(gt_paths)

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)