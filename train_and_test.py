import os
from pprint import pprint
from option import opt
pprint(opt)
import scipy.io as sio
from torch import nn
from utils_mix import generate_shift_masks_full_size
from torch.cuda.amp import GradScaler, autocast
import cv2  
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

print(os.environ["CUDA_VISIBLE_DEVICES"])

from utils_mix import *

seed_everything(
    seed = 3407,
    deterministic = True, 
)

import torch
from torch.nn.utils import clip_grad_norm_

from torch_ema import ExponentialMovingAverage
from utils_mix import generate_shift_masks_full_size

import time

from torch.profiler import profile, ProfilerActivity
import numpy as np
from torch.autograd import Variable
import datetime

from tqdm import tqdm

import losses
from schedulers import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
from architecture import *


# dataset
if opt.test_mode==0:
    train_set = LoadTraining(opt.data_path, debug=opt.debug)

if hasattr(opt, 'full_size_test') and opt.full_size_test:
    if opt.test_mode==0:
       test_data, test_file_names = LoadTest(opt.test_path, full_size=True,read_all=False)
    else:
        test_data, test_file_names = LoadTest(opt.final_test_path, full_size=True,read_all=True)  
    # Generate the mask for full-size testing.
    Phi_batch_test = generate_shift_masks_full_size(
        opt.mask_path,
        test_data.shape,
        device
    )


# Create output directories.
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
if opt.train_phase==1:
    opt.max_epoch = opt.max_epoch*2
else:
    if opt.batch_size>4:
        opt.epoch_sam_num = opt.epoch_sam_num
        
if opt.debug:
    opt.epoch_sam_num = opt.batch_size*15
elif opt.stage==9 and opt.batch_size<=2:
    opt.epoch_sam_num = opt.epoch_sam_num//2
start_epoch = 0

# model
model = nn.DataParallel(model_generator(method=opt.method,opt=opt))


if opt.train_phase==2:
    freeze_dict = dict()
    model_dict = model.state_dict()
    for param in model.parameters():
        param.requires_grad = True

    key_list = ['gt_le.']
    print('freeze key list:',end=' ')
    print(key_list)
    for (k, v) in model_dict.items():
            for kl in key_list:
                if kl in k:
                    freeze_dict[k] = v
                    
    if len(freeze_dict)>0:
        model = freeze_model(model=model, to_freeze_dict=freeze_dict)
    



# optimizing
optim_params = []
for k, v in model.named_parameters():
    if v.requires_grad:
        optim_params.append(v)
optimizer = torch.optim.Adam(optim_params, lr=opt.learning_rate, betas=(0.9, 0.999))
ema = ExponentialMovingAverage(optim_params, decay=0.999)

para_ema_sh = sum([np.prod(list(p.size())) for p in ema.shadow_params])
if ema.collected_params is not None:
    para_ema_co = sum([np.prod(list(p.size())) for p in ema.collected_params])
para_ema_up = ema.num_updates
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)), 
    num_training_steps=int(np.floor(opt.epoch_sam_num / opt.batch_size)) * opt.max_epoch, 
    eta_min=1e-6)
if opt.resume_ckpt_path:
    print("===> Loading Checkpoint from {}".format(opt.resume_ckpt_path))
    save_state = torch.load(opt.resume_ckpt_path)

    model_dict = model.state_dict()
    

    if opt.train_phase==2:
        ckpt_ema_shadow_params = save_state['ema']['shadow_params']
        ckpt_ema_shadow_params_size = sum([np.prod(list(p.size())) for p in ckpt_ema_shadow_params])
        
        print("ckpt ema size:", ckpt_ema_shadow_params_size)
        print("curr ema size:", para_ema_sh)

        ckpt_sp = ckpt_ema_shadow_params
        curr_sp = ema.shadow_params
        print("len ckpt:", len(ckpt_sp), "len curr:", len(curr_sp))

        n = min(len(ckpt_sp), len(curr_sp))
        ema_names = [k for k, v in model.named_parameters() if v.requires_grad]  

        for i in range(n):
            if tuple(ckpt_sp[i].shape) != tuple(curr_sp[i].shape):
                print("first mismatch idx:", i)
                print("param name:", ema_names[i] if i < len(ema_names) else "N/A")
                print("ckpt shape:", tuple(ckpt_sp[i].shape), "numel:", ckpt_sp[i].numel())
                print("curr shape:", tuple(curr_sp[i].shape), "numel:", curr_sp[i].numel())
                break
            
        if ckpt_ema_shadow_params_size==para_ema_sh:
            ema.load_state_dict(save_state['ema'])
        else:
            print('ema load failed')
            print("ckpt ema size:", ckpt_ema_shadow_params_size)
            print("curr ema size:", para_ema_sh)
            
        sd = save_state['model'] 
        print('miss match keys:')
        state_dict = dict()
        for k,v in sd.items():
                if ((k in model_dict.keys()) and (model_dict[k].shape==v.shape)):
                    state_dict[k] = v
                else:
                    print(k,end=',')
        

        model_dict.update(state_dict) 
        missing, unexpected = model.load_state_dict(model_dict,strict=True) 
        print(unexpected)
        print(missing)
        start_epoch = 0

    else:
        try:
            model.load_state_dict(save_state['model'])
            ema.load_state_dict(save_state['ema'])
            optimizer.load_state_dict(save_state['optimizer'])
            scheduler.load_state_dict(save_state['scheduler'])
            start_epoch = save_state['epoch']
        except:
            model_dict = model.state_dict()

            sd = save_state['model'] 
            print('miss match keys:')
            state_dict = dict()
            for k,v in sd.items():
                    if ((k in model_dict.keys()) and (model_dict[k].shape==v.shape)):
                        state_dict[k] = v
            model_dict.update(state_dict) 
            missing, unexpected = model.load_state_dict(model_dict,strict=False) 
            print(missing)
    print('\n=====')      


criterion = losses.CharbonnierLoss().to(device)

lrs = []
patch_size = opt.patch_size

print_per_layer_stat = True if opt.debug else False


flops_input_size2 = (60,128,128)
flops_input_size = (128,128)

# Log model complexity, including FLOPs and parameter count.
# If FLOPs calculation fails, parameter count is still reported.
def maybe_sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

def format_count(x):
    if x >= 1e12:
        return f"{x / 1e12:.3f} T"
    elif x >= 1e9:
        return f"{x / 1e9:.3f} G"
    elif x >= 1e6:
        return f"{x / 1e6:.3f} M"
    elif x >= 1e3:
        return f"{x / 1e3:.3f} K"
    else:
        return f"{x:.0f}"


def log_model_complexity(logger):
    total_params = sum(
        p.numel()
        for p in (model.module if isinstance(model, nn.DataParallel) else model).parameters()
    )

    try:
        net = model.module if isinstance(model, nn.DataParallel) else model
        net.eval()

        if Phi_batch_test.shape[0] == 1:
            mask = Phi_batch_test
        else:
            mask = Phi_batch_test[0:1]

        h, w = mask.shape[-2], mask.shape[-1]
        gt_dummy = torch.zeros(1, 60, h, w, device=device, dtype=torch.float32)
        meas_dummy = init_meas(gt_dummy, mask, opt.input_setting)

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        maybe_sync()
        with torch.no_grad():
            with profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_flops=True
            ) as prof:
                if opt.train_phase == 1:
                    _ = net(meas_dummy, mask, gt_dummy)
                else:
                    _ = net(meas_dummy, mask)

        maybe_sync()

        total_flops = 0
        for evt in prof.key_averages():
            if hasattr(evt, "flops") and evt.flops is not None:
                total_flops += evt.flops

        if total_flops <= 0:
            raise RuntimeError("profiler returned 0 FLOPs")

        msg = f"Model Complexity | FLOPs: {format_count(total_flops)} | Params: {total_params / 1e6:.3f} M"

    except Exception as e:
        msg = f"Model Complexity | FLOPs calculation failed: {e} | Params: {total_params / 1e6:.3f} M"

    print(msg)
    logger.info(msg)

def train(epoch, logger, scaler):
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Epoch {epoch}] GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    epoch_loss = 0

    begin = time.time()
    batch_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))
    train_tqdm = tqdm(range(batch_num))
    
    model.train()
    for i in train_tqdm:
        # Crop GT patches and keep the corresponding crop positions.
        gt_batch, crop_positions = shuffle_crop(train_set, opt.batch_size, opt.patch_size)
        gt = Variable(gt_batch).to(device)

        # Generate mask patches aligned with the cropped GT patches.
        Phi_batch_train = init_mask(
            opt.mask_path, 
            opt.input_mask, 
            opt.patch_size,
            opt.batch_size, 
            device=device,
            train_phase=opt.train_phase,
            crop_positions=crop_positions  
        )

        input_meas = init_meas(gt, Phi_batch_train, opt.input_setting)
       
        with autocast():
            model_out, log_dict = model(input_meas, Phi_batch_train, gt)
            loss = criterion(model_out, gt) 
            
            if opt.train_phase==2:
                gamma = 1.0
                prior_z = log_dict['prior_z']
                prior = log_dict['prior']
                diff_loss = criterion(prior, prior_z)
                loss = loss + gamma*diff_loss
                
        # Backpropagation with gradient scaling.
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.2)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ema.update()

        train_tqdm.set_postfix(train_loss="{:.4f}".format(loss.item()))
        epoch_loss += loss.data

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        lrs.append(lr)
        scheduler.step()
        
    end = time.time()
    train_loss = epoch_loss / batch_num
        
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} lr: {:.1f}e-4 time: {:.2f} ".
            format(epoch, train_loss, lr*10e3, (end - begin)))
    
    
    print(f"[Epoch {epoch}] GPU memory after training: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    
    return train_loss


def save_gt_out_mid5_compare_png(gt_chw: torch.Tensor,
                                 out_chw: torch.Tensor,
                                 save_path: str,
                                 num_bands: int = 5,
                                 label_w: int = 90):
    
    gt_chw = gt_chw.detach().float().cpu()
    out_chw = out_chw.detach().float().cpu()

    C, H0, W0 = gt_chw.shape
    num_bands = int(max(1, num_bands))
    start = max((C - num_bands) // 2, 0)
    end = min(start + num_bands, C)
    idxs = list(range(start, end)) if end > start else [0]
    while len(idxs) < num_bands:
        idxs.append(idxs[-1])

    gt_strips, out_strips = [], []
    H, W = None, None

    for i in idxs:
        g = gt_chw[i].transpose(0, 1)   # (H,W) -> (W,H)
        o = out_chw[i].transpose(0, 1)

        if H is None:
            H, W = g.shape

        mn = g.min()
        mx = g.max()
        den = (mx - mn).clamp_min(1e-8)
        g01 = ((g - mn) / den).clamp(0, 1)
        o01 = ((o - mn) / den).clamp(0, 1)

        gt_strips.append(g01)
        out_strips.append(o01)

    gt_img = torch.cat(gt_strips, dim=1)        # (H, num_bands*W)
    out_img = torch.cat(out_strips, dim=1)      # (H, num_bands*W)
    comp = torch.cat([gt_img, out_img], dim=0)  # (2H, num_bands*W)
    comp_u8 = (comp * 255.0).clamp(0, 255).to(torch.uint8).numpy()

    canvas = np.zeros((2 * H, label_w + num_bands * W), dtype=np.uint8)
    canvas[:, label_w:] = comp_u8

    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, y_center in (("GT", H * 0.5), ("REC", H * 1.5)):
        (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
        x = max(2, (label_w - tw) // 2)
        y = int(y_center + th // 2)
        cv2.putText(canvas, text, (x, y), font, 0.9, 255, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, canvas)

def save_meas_png(meas, save_path):
    meas = meas.detach().cpu().squeeze().numpy().astype(np.float32)  
    meas = meas - meas.min()
    meas = meas / (meas.max() + 1e-8)
    meas = (meas * 255).astype(np.uint8)
    cv2.imwrite(save_path, meas)

def test(epoch, logger, mask3d_batch_test=None, diff_test=False):
    psnr_list, ssim_list = [], []
    infer_time_list = []

    test_gt = test_data.to(torch.float32)  

    model.eval()
    begin = time.time()
    image_log = {}
    
    torch.cuda.empty_cache()

    print(f"[Epoch {epoch}] GPU memory before testing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    model_outputs = []
    #log_dicts = []
    meas_list = []  
    last_log_dict = {}  

    with torch.no_grad():
        with ema.average_parameters():
            for i in range(test_gt.shape[0]):
            
                single_gt = test_gt[i:i+1].to(device)

                
                if mask3d_batch_test is not None:
                    
                    if mask3d_batch_test.shape[0] == 1:
                        single_mask = mask3d_batch_test
                    else:
                        single_mask = mask3d_batch_test[i:i+1]
                else:
                    single_mask = None
                
                
                single_meas = init_meas(single_gt, single_mask, opt.input_setting)

                
                meas_list.append(single_meas.detach().cpu())

                
                maybe_sync()
                t0 = time.time()
                if opt.train_phase == 1:
                    single_out, single_log_dict = model(single_meas, single_mask, single_gt)
                else:
                    single_out, single_log_dict = model(single_meas, single_mask)
                maybe_sync()
                infer_time_list.append(time.time() - t0)

                
                model_outputs.append(single_out.detach().cpu())
                
                last_log_dict = single_log_dict  

                
                del single_out, single_meas, single_gt
                
                if i % 2 == 0:  
                    torch.cuda.empty_cache()
    
    
    model_out = torch.cat(model_outputs, dim=0)
    log_dict = last_log_dict

    
    print(f"\n=== Epoch {epoch} test details ===")
    for k in range(test_gt.shape[0]):
        diff = torch.abs(model_out[k] - test_gt[k])
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        
        
        if hasattr(opt, 'full_size_test') and opt.full_size_test:
            print(f"Image {k+1} ({test_file_names[k]}): PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")
        else:
            print(f"Image {k+1}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}")
    
    end = time.time()

    pred = model_out
    pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    infer_time_mean_ms = np.mean(infer_time_list) * 1000

    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, infer time = {:.2f} ms, total time: {:.2f}'
            .format(epoch, psnr_mean, ssim_mean, infer_time_mean_ms, (end - begin)))
    print(f"Average inference time per image: {infer_time_mean_ms:.2f} ms")

    psnr_str = ', '.join([f'{p:.2f}' for p in psnr_list])
    ssim_str = ', '.join([f'{s:.4f}' for s in ssim_list])

    logger.info("Test file order: " + ", ".join([os.path.basename(n) for n in test_file_names]))

    logger.info(f'PSNR: [{psnr_str}]')
    logger.info(f'SSIM: [{ssim_str}]')
    print("="*50)

    is_train = (opt.test_mode == 0)
    is_test_only = (opt.test_mode == 1)

    save_gt = (epoch == 1) or is_test_only
    save_meas_pred = is_test_only or (is_train and (epoch % 10 == 0))

    save_compare = (save_gt or save_meas_pred)

    if not (save_compare or save_meas_pred):
        return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean, image_log

    images_root = os.path.join(result_path, "images")
    os.makedirs(images_root, exist_ok=True)
    epoch_root = os.path.join(images_root, f"epoch_{epoch:03d}")
    os.makedirs(epoch_root, exist_ok=True)

    use_real_name = getattr(opt, "full_size_test", False) and ("test_file_names" in globals())

    for k in range(test_gt.shape[0]):
        if use_real_name:
            raw_name = test_file_names[k]
            base_name = os.path.splitext(os.path.basename(raw_name))[0]
        else:
            base_name = f"img_{k+1:03d}"

        sample_dir = os.path.join(epoch_root, base_name)
        os.makedirs(sample_dir, exist_ok=True)

        if save_compare:
            comp_path = os.path.join(sample_dir, "gt_out_mid5.png")
            save_gt_out_mid5_compare_png(
                gt_chw=test_gt[k],          # (C,H,W) CPU tensor
                out_chw=model_out[k],       # (C,H,W) CPU tensor
                save_path=comp_path,
                num_bands=5
            )

        if save_meas_pred:
            save_meas_png(meas_list[k], os.path.join(sample_dir, "meas.png"))
            
        # save .mat
        gt_hwc = test_gt[k].detach().float().cpu().permute(1, 2, 0).numpy().astype(np.float32)      # (H,W,C)
        rec_hwc = model_out[k].detach().float().cpu().permute(1, 2, 0).numpy().astype(np.float32)   # (H,W,C)
        sio.savemat(os.path.join(sample_dir, "gt.mat"), {"img_expand": gt_hwc})
        sio.savemat(os.path.join(sample_dir, "re.mat"), {"img_expand": rec_hwc})
        
    
    del model_outputs, last_log_dict, model_out, test_gt, meas_list
    import gc
    gc.collect()  
    torch.cuda.empty_cache()
    
    print(f"[Epoch {epoch}] GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    return pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean, image_log

def main():
    logger = gen_log(model_path)
    logger.info(opt)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))
    log_model_complexity(logger)
    scaler = GradScaler()
    psnr_max = 0

    print('init test results:')

    torch.cuda.empty_cache()

    (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, image_log) = test(0, logger, Phi_batch_test)
    if opt.test_mode==0:
        for epoch in range(start_epoch + 1, opt.max_epoch + 1):

            print(f"==>Epoch{epoch}")
            
            train_loss = train(epoch, logger, scaler)
            
            torch.cuda.empty_cache()
            
            (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean, image_log) = test(epoch, logger, Phi_batch_test)
            if psnr_mean > psnr_max:
                psnr_max = psnr_mean
                if psnr_mean > 42:
                    name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                    # scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                    checkpoint(model, ema, optimizer, scheduler, epoch, model_path, logger)
            
        
if __name__ == '__main__':
    main()


