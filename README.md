# HyperBME-Net

Official PyTorch implementation of HyperBME-Net.

---

## ⚙️ Requirements

```
python==3.10
torch==2.0.1
scikit-image==0.21.0
scikit-learn==1.5.1
numpy==1.24.4
scipy==1.11.2
pyiqa==0.1.7
matplotlib==3.7.2
Pillow==10.0.0
lpips==0.1.4
```

---

## 📂 Dataset

The full dataset and mask file are not included in this repository due to their large file size.  
Please download them manually and place them in the following structure.

- Dataset:https://pan.baidu.com/s/1l975wOK5E-5ieS_8e4HzIA?pwd=fpkk
- Mask file:https://pan.baidu.com/s/1-kp-qXgSs16RKaY8vs-VyQ?pwd=2ps6

Expected structure:

```
datasets/
└── train_test/
    ├── train/
    │   └── *.mat
    ├── test/
    │   └── *.mat
    ├── final_test/
    │   └── *.mat
    └── Mask_HyperspecI_V1.mat
```

Here, `test/` is used as the validation set, while `final_test/` is used as the final test set.
---

## 📁 Project Structure

```
HyperBME-Net/
├── architecture/        # network architecture
├── datasets/            # dataset structure
├── exp/                 # experiment outputs and checkpoints
├── options/             # configuration files
├── ptflops/             # model complexity tools
├── train_and_test.py    # training and testing script
├── losses.py            # loss functions
├── schedulers.py        # learning rate scheduler
└── utils_mix.py         # utility functions
```

---

## 🚀 Training & Testing

### Training

#### 1st Phase

```
python train_and_test.py --train_phase 1  --clip_grad --batch_size 64 --template hyper --outf ./exp/hyper__1st/ --method hyper --stage 3 --body_share_params 0 
```
### Debug Mode

You can add `--debug 1` to quickly check whether the code runs correctly with a small amount of data.

Example:

```
python train_and_test.py --train_phase 1  --clip_grad --batch_size 64 --template hyper --outf ./exp/hyper__1st/ --method hyper --stage 3 --body_share_params 0 --debug 1
```

#### 2nd Phase

Please modify `--resume_ckpt_path` before running the second training phase.

```
python train_and_test.py --train_phase 2  --clip_grad --batch_size 56 --template hyper --outf ./exp/hyper_2nd/ --resume_ckpt_path ./exp/hyper_1st/model/model_epoch_xxx.pth --method hyper --stage 3 --body_share_params 0 
```

### Testing

Please modify `--resume_ckpt_path` before testing.

```
python train_and_test.py --train_phase 2  --test_mode 1  --template hyper --outf ./exp/hyper_test/ --method hyper --stage 3 --body_share_params 0 --resume_ckpt_path ./exp/hyper_2nd/model/model_epoch_xxx.pth --gpu_id 1
```

---

## ⚠️ Notes

- Please place the `.mat` files according to the expected dataset structure.
- The `exp/` folder is used to save training logs, checkpoints, and testing results.

