import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
import h5py
import mat73
import cv2
from tqdm import tqdm

def freeze_model(model, to_freeze_dict, keep_step=None):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass
    return model

def generate_shift_masks(mask_path, patch_size, batch_size, device, train_phase=0, crop_positions=None):
    """
        Generate mask patches for training or testing.

        If crop positions are provided, the mask is cropped consistently with the
        corresponding HSI patch.
    """
    mat = sio.loadmat(mask_path + '/Mask_HyperspecI_V1.mat')
    mask_3d_shift = np.array(mat['mask']) 
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    if crop_positions is None:
        Phi_batch = get_mask_patches_fixed_center(mask=mask_3d_shift, 
                                                 image_size=2048, 
                                                 patch_size=patch_size, 
                                                 batch_size=batch_size)
    else:
        Phi_batch = get_mask_patches_with_positions(mask=mask_3d_shift,
                                                   patch_size=patch_size,
                                                   crop_positions=crop_positions)
    
    Phi_batch = Phi_batch.to(torch.float32).to(device)
    return Phi_batch

def get_mask_patches_fixed_center(mask, image_size, patch_size, batch_size):
    fixed_h = (image_size - patch_size) // 2
    fixed_w = (image_size - patch_size) // 2

    masks = []
    for _ in range(batch_size):
        mask_patch = mask[:, 
                          fixed_h : fixed_h + patch_size, 
                          fixed_w : fixed_w + patch_size]
        mask_patch = np.maximum(mask_patch, 0)
        mask_patch = mask_patch / mask_patch.max()
        masks.append(mask_patch)

    mask_patches = torch.stack(masks, dim=0)
    return mask_patches

def get_mask_patches_with_positions(mask, patch_size, crop_positions):
    """Crop mask patches according to the given crop positions."""
    masks = []
    for crop_pos in crop_positions:
        """Create a mosaic mask from four cropped mask patches."""
        if isinstance(crop_pos, list):
            mask_patch = create_mosaic_mask(mask, patch_size, crop_pos)
        else:
            h_start, w_start = crop_pos
            mask_patch = mask[:, 
                              h_start : h_start + patch_size,
                              w_start : w_start + patch_size]
            mask_patch = np.maximum(mask_patch, 0)
            mask_patch = mask_patch / mask_patch.max()
        masks.append(mask_patch)
    
    mask_patches = torch.stack(masks, dim=0)
    return mask_patches

def generate_shift_masks_full_size(mask_path, test_data_shape, device):

    mat = sio.loadmat(mask_path + '/Mask_HyperspecI_V1.mat')
    mask_3d_shift = np.array(mat['mask']) 
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    
    batch_size, nC, H, W = test_data_shape
    
    Phi_batch = torch.clamp(mask_3d_shift, min=0)
    Phi_batch = Phi_batch / Phi_batch.max()
    
    Phi_batch = Phi_batch.to(torch.float32).to(device).unsqueeze(0)

    print(f"[generate_shift_masks_full_size] Phi.shape = {Phi_batch.shape}")
    return Phi_batch

def create_mosaic_mask(mask, patch_size, crop_positions_list):

    half_size = patch_size // 2
    nC = mask.shape[0]
    
    output_mask = torch.zeros(nC, patch_size, patch_size)
    
    for i, (h_start, w_start) in enumerate(crop_positions_list):
        
        small_patch = mask[:, 
                          h_start : h_start + half_size,
                          w_start : w_start + half_size]
        small_patch = np.maximum(small_patch, 0)
        small_patch = small_patch / small_patch.max()
        
        if i == 0:  # top-left
            output_mask[:, :half_size, :half_size] = small_patch
        elif i == 1:  # top-right
            output_mask[:, :half_size, half_size:] = small_patch
        elif i == 2: # bottom-left
            output_mask[:, half_size:, :half_size] = small_patch
        elif i == 3:  # bottom-right
            output_mask[:, half_size:, half_size:] = small_patch
    
    return output_mask

def LoadTraining(path, debug=False, to_tensor=False):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training scenes:', len(scene_list))
    num_scenes = len(scene_list) if not debug else min(3, len(scene_list))
    for i in tqdm(range(num_scenes), desc="Loading training scenes"):
        scene_path = path + scene_list[i]
        scene_num = int(scene_list[i].split('.')[0][5:])
        if scene_num<=205:
            if 'mat' not in scene_path:
                continue
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            
            if to_tensor:
                img = torch.from_numpy(img)
                img = torch.clamp(img, 0.0, 1.0)
            else:
                img = img.astype(np.float32)
            imgs.append(img)

    return imgs

def LoadTest(path_test, full_size=False,read_all=False):
    """Load test data as full-size images or random 128x128 crops."""
    all_files = os.listdir(path_test)
    mat_files = sorted([f for f in all_files if f.endswith('.mat')])
    
    
    if full_size:
        selected_files = mat_files if read_all else random.sample(mat_files, 10)

        # Load one image to get the full spatial size.
        first_file_path = os.path.join(path_test, selected_files[0])
        first_img = sio.loadmat(first_file_path)['img_expand']
        H, W, C = first_img.shape
        
        N = len(selected_files)  # [MOD]
        test_data = np.zeros((N, H, W, C), dtype=np.float32)  # [MOD]
        test_crop_positions = None  
        
        
        for i, file_name in enumerate(selected_files):
            file_path = os.path.join(path_test, file_name)
            img = sio.loadmat(file_path)['img_expand']
            test_data[i, :, :, :] = img
            
    else:
        test_data = np.zeros((10, 128, 128, 60))
        test_crop_positions = []
        
        
        for i, file_name in enumerate(selected_files):
            file_path = os.path.join(path_test, file_name)
            img = sio.loadmat(file_path)['img_expand']
            
            h_start = random.randint(0, img.shape[0] - 128 - 1)
            w_start = random.randint(0, img.shape[1] - 128 - 1)
            test_crop_positions.append((h_start, w_start))
            
            cropped_img = img[h_start:h_start+128, w_start:w_start+128, :]
            test_data[i, :, :, :] = cropped_img
    
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2))) / 65535
    test_data = torch.clamp(test_data, 0.0, 1.0)
    
    if full_size:
        return test_data, selected_files  
    else:
        return test_data, test_crop_positions
    

def torch_psnr(img, ref):
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i] - ref[i])**2)
        #print(f"[DEBUG] channel {i}: mse = {mse.item()}")
        psnr += 10 * torch.log10((255*255) / mse)
    return psnr / nC

def torch_ssim(img, ref):  
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def shuffle_crop(train_data, batch_size, crop_size=128, augment=True):
    """Randomly crop training patches and return their crop positions."""
    crop_positions = []  
    
    if augment:
        flag = random.randint(0, 1)
        if flag:
            index = np.random.choice(range(len(train_data)), batch_size)
            processed_data = np.zeros((batch_size, crop_size, crop_size, 60), dtype=np.float32)
            for i in range(batch_size):
                h, w, _ = train_data[index[i]].shape
                x_index = np.random.randint(0, h - crop_size)
                y_index = np.random.randint(0, w - crop_size)
                crop_positions.append((x_index, y_index))  
                processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
            gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
            for i in range(gt_batch.shape[0]):
                gt_batch[i] = augment_1(gt_batch[i])
        else:
            gt_batch = []
            for i in range(batch_size):
                sample_list = np.random.randint(0, len(train_data), 4)
                processed_data = np.zeros((4, crop_size//2, crop_size//2, 60), dtype=np.float32)
                batch_crop_positions = []  
                for j in range(4):
                    h, w, _ = train_data[sample_list[j]].shape
                    x_index = np.random.randint(0, h-crop_size//2)
                    y_index = np.random.randint(0, w-crop_size//2)
                    batch_crop_positions.append((x_index, y_index))
                    processed_data[j] = train_data[sample_list[j]][x_index:x_index+crop_size//2,y_index:y_index+crop_size//2,:]
                crop_positions.append(batch_crop_positions)
                generated_sample = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))
                gt_batch.append(augment_2(generated_sample))
            gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch, crop_positions
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size, crop_size, crop_size, 60), dtype=np.float32)
        for i in range(batch_size):
            h, w, _ = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            crop_positions.append((x_index, y_index))
            processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + crop_size, y_index:y_index + crop_size, :]
        gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))

    return gt_batch, crop_positions

def augment_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def augment_2(generate_gt):
    bs,c,h,w = generate_gt.shape
    h = h*2
    w = w*2

    divid_point_h = h//2
    divid_point_w = w//2
    output_img = torch.zeros(c,h,w)
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img

def gen_meas_torch(data_batch, Phi_batch):
    [batch_size, nC, H, W] = data_batch.shape 
    meas = torch.sum(data_batch*Phi_batch, 1) 
    meas = meas / nC * 2
    return meas
    
def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def init_mask(mask_path, mask_type, patch_size, batch_size, device="cuda", train_phase=0, crop_positions=None):
    """Initialize sensing masks and optionally align them with crop positions."""
    if mask_type == 'Phi':
        Phi_batch = generate_shift_masks(mask_path, patch_size, batch_size, device, train_phase, crop_positions)
    elif mask_type == 'Mask':
        Phi_batch = generate_masks(mask_path, batch_size)
        # Phi_batch = mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size, device, crop_positions=crop_positions)
        input_mask = (Phi_batch, Phi_s_batch)
        return Phi_batch, input_mask
    return Phi_batch

def input_with_mask(image, prob_=0.70, value=0.1):

    bs,x,y = image.shape
    
    # Generate mask on GPU
    mask = torch.bernoulli(torch.full((bs,x, y), prob_, device='cuda'))
    # mask = mask.unsqueeze(2).expand(x, y, nc)  # Expand to match image channels
    
    # Apply mask and noise on GPU
    noise_image = image * mask
    noise_image = noise_image - value + value * mask
    
    return noise_image

def init_meas(gt, phi, input_setting, opt=None):
    if input_setting == 'Y':
        input_meas = gen_meas_torch(gt, phi)
        
    if opt is not None:
        if opt.mask_input:
            input_meas = input_with_mask(input_meas)
    return input_meas

def checkpoint(model, ema, optimizer, scheduler,  epoch, model_path, logger):
    save_dict = {}
    save_dict['model'] = model.state_dict()
    save_dict['ema'] = ema.state_dict()
    save_dict['optimizer'] = optimizer.state_dict()
    save_dict['scheduler'] = scheduler.state_dict()
    save_dict['epoch'] = epoch
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(save_dict, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
    
def checkpoint_simple(model, optimizer,  epoch, model_path, logger):
    save_dict = {}
    save_dict['model'] = model.state_dict()

    save_dict['optimizer'] = optimizer.state_dict()

    save_dict['epoch'] = epoch
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(save_dict, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def freeze_model(model, to_freeze_dict, keep_step=None):
    print('freeze_dict:',end=' ')  
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
            print(name,end=', ')
        else:
            param.requires_grad = True
            pass
    print('\n=====')
    return model

def seed_everything(
    seed = 3407,
    deterministic = False, 
):
    """Set random seed.
    Args:
        seed (int): Seed to be used, default seed 3407, from the paper
        Torch. manual_seed (3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision[J]. arXiv preprint arXiv:2109.08203, 2021.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False