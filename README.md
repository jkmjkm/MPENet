# MPENet 

[//]: # (### 📖[**Paper**]&#40;https://ieeexplore.ieee.org/document/10387229&#41; | 🖼️[**PDF**]&#40;/fig/TTST.pdf&#41;)

PyTorch codes for "MPENet: Multi-Prior Enhancement Network for FY-4B Geostationary Satellite Imagery Super-Resolution"

- Authors: Kunming Jiang, Jingwen Yuan, [Yi Xiao*](https://xy-boy.github.io/), [Tingting Liu*](https://pole.whu.edu.cn/cn/scholar/14.html), [Qiangqiang Yuan](https://jszy.whu.edu.cn/yuanqiangqiang)<br>
## Abstract
> Geostationary satellite imagery plays a critical role in meteorological observation and severe weather forecasting. However, the ultra-long imaging distance and hardware constraints result in severe spatial resolution degradation with weak textures and blurred boundaries. Existing video super-resolution methods primarily focus on motion compensation, overlooking the illumination variations between observations. To address these challenges, we propose Multi-Prior Enhancement Network (MPENet) for Geostationary Satellite Super-Resolution that performs adaptive calibration from three complementary perspectives: Frequency Statistics Guided Modulation (FSGM) perceives global illumination fluctuations via frequency-domain statistics, Energy-Structure Aware Enhancement (ESAE) captures illumination-direction-induced variations through energy-based gating, and Local Texture Enhancement (LTE) compensates for high-frequency detail loss via multi-directional learnable gradient convolutions. Furthermore, a Fourier Residual Block (FRB) is incorporated to enhance frequency-domain modeling with global receptive fields while introducing negligible parameter overhead. Extensive experiments on our FY-4B satellite dataset demonstrate that MPENet consistently outperforms state-of-the-art SISR and VSR methods, achieving a PSNR of 33.26 dB while maintaining competitive computational costs.
## Network  
 ![image](/fig/network.png)
 
## 🧩 Install
```
git clone https://github.com/jkmjkm/MPENet.git
```

## Environment
 > * CUDA 11.1
 > * Python 3.9.23
 > * PyTorch 1.9.1
 > * Torchvision 0.10.1
 > * basicsr 1.4.2 

## 🎁 Dataset
The dataset for training and testing is available at [this link]&#40;https://pan.quark.cn/s/4574669f4922?pwd=4Hf7&#41;.<br>
### Data Source
This dataset is derived from the Level-1 (L1) data of the **Fengyun-4B (FY-4B)** satellite,
which is publicly shared by the **National Satellite Meteorological Center (NSMC)** of China.
- Official data portal: [http://satellite.nsmc.org.cn/](http://satellite.nsmc.org.cn/)
- The raw L1 data was converted to PNG images and cropped into region patches to form this dataset.
 ![image](/fig/data.png)

## 🧩 Usage
### Quick Test
[Download Pre-trained Model](https://github.com/jkmjkm/MPENet/blob/master/experiments/pretrained_models/net_g_145000.pth)
### Option 1: Quick Test with Provided Sample Data
For fast testing with our pre-packaged sample dataset, simply run the following command:

```bash
python eval_4x.py
```
### Option 2: Test with Your Own Dataset
- **Step I.**  Use the structure below to prepare your dataset.

/xxxx/xxx/ (your data path)
```
/GT/ 
   /0000.png  
   /····.png  
   /0007.png  
/LR/ 
   /0000.png  
   /····.png  
   /0007.png  
```
- **Step II.**  Change the `dataroot_gt` and `dataroot_lq` in `options/test/MPENet/test_MPENet_FY.yml` to your data path.
- **Step III.**  Run the [test.py](https://github.com/jkmjkm/MPENet/blob/master/basicsr/test.py)
```
python basicsr/test.py -opt options/test/MPENet/test_MPENet_FY.yml
```
### Train
Run the [train.py](https://github.com/jkmjkm/MPENet/blob/master/basicsr/train.py)
```
python basicsr/train.py -opt options/train/MPENet/train_MPENet_FY.yml
```

## 🖼️ Results
### Quantitative
 ![image](/fig/compare.png)

### Visual
 ![image](/fig/visual.png)

## Acknowledgments
Our MPENet mainly borrows from [MADNet](https://github.com/XY-boy/MADNet) and [FourierSR](https://github.com/PRIS-CV/FourierSR). Thanks for these excellent open-source works!

## Contact
If you have any questions or suggestions, feel free to contact me.  
Email: kunmingjiang@whu.edu.cn

## Citation
If you find our work helpful in your research, please consider citing it. Your support is greatly appreciated! 😊

[//]: # (```)

[//]: # (@ARTICLE{xiao2024ttst,)

[//]: # (  author={Xiao, Yi and Yuan, Qiangqiang and Jiang, Kui and He, Jiang and Lin, Chia-Wen and Zhang, Liangpei},)

[//]: # (  journal={IEEE Transactions on Image Processing}, )

[//]: # (  title={TTST: A Top-k Token Selective Transformer for Remote Sensing Image Super-Resolution}, )

[//]: # (  year={2024},)

[//]: # (  volume={33},)

[//]: # (  number={},)

[//]: # (  pages={738-752},)

[//]: # (  doi={10.1109/TIP.2023.3349004})

[//]: # (})

[//]: # (```)
