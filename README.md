# Unleashing the Power of Gradient Signal-to-Noise Ratio for Zero-Shot NAS (ICCV 2023)
![license](https://img.shields.io/badge/License-MIT-brightgreen)
![python](https://img.shields.io/badge/Python-3.7-blue)
![pytorch](https://img.shields.io/badge/PyTorch-1.1-orange)

This is an official pytorch implementation for ([Unleashing the Power of Gradient Signal-to-Noise Ratio for Zero-Shot NAS](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Unleashing_the_Power_of_Gradient_Signal-to-Noise_Ratio_for_Zero-Shot_NAS_ICCV_2023_paper.pdf)) by Zihao Sun, and et. al.

## Requirements
Please firstly create a virtual environment and install the folowing requirements
```shell
pip install -r pip_requirements.txt
conda install --yes --file conda_requirements.txt
```
Then, install nasbench
```shell
cd nasbench
pip install -e .
```

## Dataset
Please download the datasets and nasbench from Google Cloud Drive, and put them in the corresponding folders.

[_dataset](https://drive.google.com/drive/folders/1_SFH7aJzFXWMXR-PvUYh9djNtB91VJ_s)

[201_api](https://drive.google.com/drive/folders/1AOM8_e7gx3D54dJicIYnUUS_sFiuMO4p)

[101_api](https://drive.google.com/drive/folders/1Q0EHqqYPQAaV1b8GT_9IeBbboAijtx_K)

[nds_data](https://drive.google.com/drive/folders/1rlU4ueIZYFrw3XSzhLPfgq2aSe6Vfrc7)


## Usage

### AGNAS in Darts_Search_Space

> Xi_GSNR Consistency in NAS-Bench-201 search space.
```shell
- Bench201-cifar10
python run_gsnr.py --api_loc='../201_api/NAS-Bench-201-v1_0-e61699.pth' --nasspace='nasbench201' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_bench201_c10_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

- Bench201-cifar100
python run_gsnr.py --api_loc='../201_api/NAS-Bench-201-v1_0-e61699.pth' --nasspace='nasbench201' --data_loc='../_dataset/cifar100/' --dataset='cifar100' --num_classes=100 --GPU='0' --save='./logs/gsnr_bench201_c100_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

- Bench201-ImageNet16-120
python run_gsnr.py --api_loc='../201_api/NAS-Bench-201-v1_0-e61699.pth' --nasspace='nasbench201' --data_loc='../_dataset/imagenet_16_120/ImageNet16/' --dataset='ImageNet16-120' --num_classes=120 --GPU='0' --save='./logs/gsnr_bench201_im16_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08
```

> Xi_GSNR Consistency in NAS-Bench-101 search space.
```shell
python run_gsnr.py --api_loc='../101_api/nasbench_only108.tfrecord' --nasspace='nasbench101' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_bench101_c10_test.log' --end=4500 --batch_size=8 --batch_numbers=8 --random_xi=1e-08
```

> Xi_GSNR Consistency in NDS search space.
```shell
python run_gsnr.py --api_loc='../nds_data' --nasspace='nds_darts' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_nds_darts_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

python run_gsnr.py --api_loc='../nds_data' --nasspace='nds_enas' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_nds_enas_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

python run_gsnr.py --api_loc='../nds_data' --nasspace='nds_pnas' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_nds_pnas_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

python run_gsnr.py --api_loc='../nds_data' --nasspace='nds_darts' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_nds_darts_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08

python run_gsnr.py --api_loc='../nds_data' --nasspace='nds_nasnet' --data_loc='../_dataset/cifar10/' --dataset='cifar10' --num_classes=10 --GPU='0' --save='./logs/gsnr_nds_nasnet_test.log' --end=0 --batch_size=64 --batch_numbers=8 --random_xi=1e-08
```

## Citation
Please cite our paper if you find anything helpful.
```
@inproceedings{sun2023unleashing,
  title={Unleashing the power of gradient signal-to-noise ratio for zero-shot NAS},
  author={Sun, Zihao and Sun, Yu and Yang, Longxing and Lu, Shun and Mei, Jilin and Zhao, Wenxiao and Hu, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5763--5773},
  year={2023}
}
```






