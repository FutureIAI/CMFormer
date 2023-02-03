# CMFormer

## Experiment result

Results on NYU Depth V2 dataset

|   model    | MPA(%) | mIoU(%) | Params/M | FLOPs/G |                            Weight                            |
| :--------: | :----: | :-----: | :------: | :-----: | :----------------------------------------------------------: |
| CMFormer-S | 62.01  |  50.42  |  47.93   |  35.97  | [CMFormer-S](链接：https://pan.baidu.com/s/1TzHuv3u0JJYGsH5C-5Tc4w?pwd=CDUT <br/>提取码：CDUT) |
| CMFormer-M | 66.36  |  54.12  |  81.75   |  68.38  | [CMFormer-M](链接：https://pan.baidu.com/s/189TVQZF59ZXjUs7ZxR62LA?pwd=CDUT <br/>提取码：CDUT) |
| CMFormer-L | 68.00  |  55.75  |  131.41  | 128.92  | [CMFormer-L](链接：https://pan.baidu.com/s/1RYB8Jk0la8irm3UNz9KBzA?pwd=CDUT <br/>提取码：CDUT) |

### Requirements

```
python3
timm
mmsegmentation
mmcv
einops
ml_collections
pytorch==1.8.2+cu111
```

Download the pre-training weight of pvt_v2([PVT/classification at v2 · whai362/PVT · GitHub](https://github.com/whai362/PVT/tree/v2/classification))

## How to use

Modify the configuration in file `get_config.py` and run `train.py` or `eval.py`

## Note

The code is partially based on ACNet([GitHub - anheidelonghu/ACNet: ACNet: Attention Complementary Network for RGBD semantic segmentation](https://github.com/anheidelonghu/ACNet)) and mmsegmentation([GitHub - open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation)) 

