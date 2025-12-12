# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import logging

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from contiformer import ContiFormer
from dataset_timeSeries import TimeSeriesDataset_Interpolation_roundedInput
from torch.utils.data import DataLoader
from tqdm import tqdm


class TqdmCompatibleHandler(logging.StreamHandler):
    """Logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# matplotlib.use('agg')


def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    # Clear any pre-existing handlers so we do not get duplicate lines when Hydra/root logging is configured.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = TqdmCompatibleHandler()
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    return logger


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



# class ContiFormer(nn.Module):
#     """
#     Improved ContiFormer for time series denoising and interpolation.

#     Key improvements:
#     - Multi-layer encoder (configurable via n_layers)
#     - Configurable model dimensions (d_model, d_inner, n_head)
#     - Layer normalization for stable training
#     - Optional smoothness and frequency-domain losses
#     - Better device management
#     """

#     def __init__(self, obs_dim, device, cfg):
#         super(ContiFormer, self).__init__()

#         # Get configurable architecture parameters with sensible defaults
#         self.d_model = getattr(cfg, 'd_model', 64)
#         d_inner = getattr(cfg, 'd_inner', 256)
#         n_layers = getattr(cfg, 'n_layers', 3)
#         n_head = getattr(cfg, 'n_head', 4)
#         d_k = self.d_model // n_head
#         d_v = self.d_model // n_head
#         actfn = getattr(cfg, 'actfn', 'softplus')

#         # ODE solver configuration
#         args_ode = AttrDict({
#             'use_ode': True,
#             'actfn': actfn,
#             'layer_type': 'concat',
#             'zero_init': True,
#             'atol': cfg.atol,
#             'rtol': cfg.rtol,
#             'method': cfg.method,
#             'regularize': False,
#             'approximate_method': 'gauss',
#             'nlinspace': 3,
#             'linear_type': 'before',
#             'interpolate': 'linear',
#             'itol': 1e-3
#         })

#         # Input/output projections
#         self.lin_in = nn.Linear(obs_dim, self.d_model)
#         self.lin_out = nn.Linear(self.d_model, obs_dim)

#         # Multi-layer encoder stack
#         self.encoder_layers = nn.ModuleList([
#             EncoderLayer(self.d_model, d_inner, n_head, d_k, d_v,
#                         args=args_ode, dropout=cfg.dropout)
#             for _ in range(n_layers)
#         ])

#         # Final layer normalization for stability
#         self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

#         # Positional encoding vector
#         self.position_vec = torch.tensor(
#             [math.pow(10000.0, 2.0 * (i // 2) / self.d_model)
#              for i in range(self.d_model)]
#         )

#         self.batch_size = cfg.batch_size
#         self.device = device

#         # Loss tracking
#         self.L1_loss = []
#         self.gradient_loss = []
#         self.smooth_loss = []
#         self.total_loss = []

#         # Loss weights from config
#         self.weight_l1 = getattr(cfg, 'weight_l1', 1.0)
#         self.weight_grad = getattr(cfg, 'weight_grad', 0.5)
#         self.weight_smooth = getattr(cfg, 'weight_smooth', 0.1)
#         self.use_frequency_loss = getattr(cfg, 'use_frequency_loss', False)

#         # Move model to device
#         self.to(device)

#     def temporal_enc(self, time):
#         """
#         Sinusoidal temporal encoding.
#         Input: batch*seq_len.
#         Output: batch*seq_len*d_model.
#         """
#         result = time.unsqueeze(-1) / self.position_vec.to(time.device)
#         result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
#         result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
#         return result

#     def pad_input(self, input, t0, tmax=None):
#         """Pad input sequence for interpolation boundary handling."""
#         input_last = input[:, -1:, :]
#         input = torch.cat((input, input_last), dim=1)
#         if tmax is None:
#             if len(t0) > 1:
#                 dt = torch.clamp(t0[-1] - t0[-2], min=1e-3)
#             else:
#                 dt = torch.tensor(1e-3, device=t0.device)
#             tmax = t0[-1] + dt
#         t0 = torch.cat((t0, tmax.unsqueeze(0)), dim=0)
#         return input, t0

#     def forward(self, samples, orig_ts, **kwargs):
#         is_train = kwargs.get('is_train', False)
#         bs, ls = samples.shape[0], len(orig_ts)

#         # Subsample batch during training for efficiency
#         if is_train:
#             actual_batch = min(self.batch_size, bs)
#             sample_idx = torch.from_numpy(
#                 npr.choice(bs, actual_batch, replace=False)
#             ).long().to(samples.device)
#             samples = samples[sample_idx, ...]
#             bs = actual_batch
#         else:
#             sample_idx = None

#         # Extract time and project input
#         t0 = samples[..., -1]
#         x = self.lin_in(samples[..., :-1])
#         x = (x + self.temporal_enc(t0)).float()

#         # Interpolate to target timestamps using CDE
#         _x, _t0 = self.pad_input(x, t0[0])
#         X = torchcde.LinearInterpolation(_x, t=_t0)
#         x = X.evaluate(orig_ts).float()

#         orig_ts = torch.as_tensor(orig_ts, device=x.device).float()
#         t_broadcast = orig_ts.unsqueeze(0).repeat(bs, 1).float()

#         # Attention mask (all visible for denoising)
#         mask = torch.zeros(bs, ls, 1, device=x.device).bool()

#         # Apply multi-layer encoder
#         for layer in self.encoder_layers:
#             x, _ = layer(x, t_broadcast, mask=mask)

#         # Apply final layer norm and output projection
#         x = self.layer_norm(x)
#         out = self.lin_out(x)

#         return out, sample_idx

# def calculate_loss(pred_x, target_x, cfg, time_interval=None,
#                        weight_l1=None, weight_grad=None, weight_smooth=None, idx=None):
#     """
#     Calculate combined loss for denoising and interpolation.

#     Includes:
#     - Smooth L1 loss for reconstruction
#     - Gradient matching loss for preserving dynamics
#     - Smoothness regularization for denoising
#     - Optional frequency domain loss
#     """
#     # Use instance weights if not provided
#     # weight_l1 = weight_l1 if weight_l1 is not None else self.weight_l1
#     # weight_grad = weight_grad if weight_grad is not None else self.weight_grad
#     # weight_smooth = weight_smooth if weight_smooth is not None else self.weight_smooth

#     if idx is not None:
#         target_x = target_x[idx, ...]

#     # Primary reconstruction loss
#     l1_loss = torch.nn.functional.smooth_l1_loss(pred_x, target_x, reduction="mean")

#     # Gradient matching loss
#     if time_interval is not None and len(time_interval) > 1:
#         dt = (time_interval[1:] - time_interval[:-1]).unsqueeze(-1)
#         dt = torch.clamp(dt, min=1e-6)  # Avoid division by zero
#         pred_grad = (pred_x[:, 1:] - pred_x[:, :-1]) / dt
#         target_grad = (target_x[:, 1:] - target_x[:, :-1]) / dt
#         gradient_loss = torch.nn.functional.mse_loss(pred_grad, target_grad, reduction="mean")
#     else:
#         gradient_loss = torch.tensor(0.0, device=pred_x.device)

#     # Smoothness regularization (penalize 2nd derivative for denoising)
#     if pred_x.shape[1] > 2:
#         second_deriv = pred_x[:, 2:] - 2 * pred_x[:, 1:-1] + pred_x[:, :-2]
#         smooth_loss = torch.mean(second_deriv ** 2)
#     else:
#         smooth_loss = torch.tensor(0.0, device=pred_x.device)

#     # Combine losses
#     total_loss = (weight_l1 * l1_loss +
#                     weight_grad * gradient_loss +
#                     weight_smooth * smooth_loss)

#     # Optional frequency domain loss for better denoising
#     use_frequency_loss = getattr(cfg, 'use_frequency_loss', False)
#     if use_frequency_loss and pred_x.shape[1] >= 8:
#         pred_fft = torch.fft.rfft(pred_x.squeeze(-1), dim=1)
#         target_fft = torch.fft.rfft(target_x.squeeze(-1), dim=1)
#         freq_loss = torch.mean(torch.abs(pred_fft - target_fft))
#         total_loss = total_loss + 0.1 * freq_loss

#     # Track losses
#     self.L1_loss.append(l1_loss.item())
#     self.gradient_loss.append(gradient_loss.item())
#     self.smooth_loss.append(smooth_loss.item())
#     self.total_loss.append(total_loss.item())

#     return total_loss, l1_loss, gradient_loss


def get_ds_timeSeries(cfg):
    train_count = cfg.train_count
    val_count = cfg.val_count
    x_values = np.arange(0, cfg.number_x_values)

    train_ds = TimeSeriesDataset_Interpolation_roundedInput(train_count, x_values, cfg)
    val_ds = TimeSeriesDataset_Interpolation_roundedInput(val_count, x_values, cfg)

    train_dataloader = DataLoader(train_ds, batch_size=cfg.batch_size)
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader


def setup_environment(cfg):
    os.makedirs(cfg.train_dir, exist_ok=True)
    os.makedirs(cfg.val_dir_pictures, exist_ok=True)
    os.makedirs(cfg.val_dir_data, exist_ok=True)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() and cfg.gpu >= 0 else 'cpu')
    log = get_logger(os.path.join(cfg.train_dir, 'log'))
    return device, log


def build_model_and_optimizer(cfg, device, log):
    obs_dim = 1
    model = ContiFormer(input_size=obs_dim,
                        d_model=getattr(cfg, 'd_model', 256),
                        d_inner=getattr(cfg, 'd_inner', 256),
                        n_layers=getattr(cfg, 'n_layers', 3),
                        n_head=getattr(cfg, 'n_head', 4),
                        d_k=getattr(cfg, 'd_k', 64),
                        d_v=getattr(cfg, 'd_v', 64),
                        dropout=getattr(cfg, 'dropout', 0.1),
                        actfn_ode=getattr(cfg, 'actfn', "softplus"),
                        layer_type_ode=getattr(cfg, 'layer_type_ode', "concat"),
                        zero_init_ode=getattr(cfg, 'zero_init_ode', True),
                        atol_ode=getattr(cfg, 'atol', 1e-6),
                        rtol_ode=getattr(cfg, 'rtol', 1e-6),
                        method_ode=getattr(cfg, 'method', "rk4"),
                        linear_type_ode=getattr(cfg, 'linear_type_ode', "inside"),
                        regularize=getattr(cfg, 'regularize', 256),
                        approximate_method=getattr(cfg, 'approximate_method', "last"),
                        nlinspace=getattr(cfg, 'nlinspace', 3),
                        interpolate_ode=getattr(cfg, 'interpolate_ode', "linear"),
                        itol_ode=getattr(cfg, 'itol_ode', 1e-2),
                        add_pe=getattr(cfg, 'add_pe', False),
                        normalize_before=getattr(cfg, 'normalize_before', False),
                        # max_length=getattr(cfg, 'max_length', 100),
                        )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = None
    start_itr = 0

    # Build learning rate scheduler
    if getattr(cfg, 'use_scheduler', False):
        scheduler_type = getattr(cfg, 'scheduler_type', 'cosine')
        warmup_epochs = getattr(cfg, 'warmup_epochs', 100)
        min_lr = getattr(cfg, 'min_lr', 1e-6)

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.niters - warmup_epochs, eta_min=min_lr
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.niters // 4, gamma=0.5
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=50, min_lr=min_lr
            )
        log.info(f'Using {scheduler_type} learning rate scheduler')

    if cfg.train_dir is not None:
        ckpt_path = os.path.join(cfg.train_dir, f'ckpt_{cfg.model_name}.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_itr = checkpoint['itr']
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log.info('Loaded ckpt from {}'.format(ckpt_path))

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Model: {cfg.model_name} | Total params: {total_params:,} | Trainable: {trainable_params:,}')
    log.info(f'Architecture: d_model={getattr(cfg, "d_model", 64)}, n_layers={getattr(cfg, "n_layers", 3)}, n_head={getattr(cfg, "n_head", 4)}')

    return model, optimizer, scheduler, start_itr


def save_checkpoint(model, optimizer, scheduler, itr, cfg, log, best=False):
    if cfg.train_dir is None:
        return
    suffix = '_best' if best else ''
    ckpt_path = os.path.join(cfg.train_dir, f'ckpt_{cfg.model_name}{suffix}.pth')
    save_dict = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'itr': itr,
    }
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, ckpt_path)
    # log.info('Stored ckpt at {}'.format(ckpt_path))


def train_one_epoch(model, optimizer, train_loader, device, epoch, loss_meter, cfg, log):
    model.train()
    iterable = train_loader
    total = len(train_loader)
    epoch_loss = 0.0
    num_batches = 0

    if not cfg.disable_tqdm:
        iterable = tqdm(train_loader, total=total, dynamic_ncols=True,
                        desc=f'Training {epoch:04d}', leave=False)

    for batch_idx, batch in enumerate(iterable, start=1):
        optimizer.zero_grad()

        groundTruth = batch["groundTruth"].unsqueeze(-1).to(device)
        timeSeries_noisy_original = batch["noisy_TimeSeries"].to(device)
        mask = batch["mask"].to(device)
        # time_stamps_original = batch["time_stamps"].to(device)

        # mask_indices = torch.where(mask[0] == True)[0]
        # timeSeries_noisy = timeSeries_noisy_original[:, mask_indices].unsqueeze(-1)
        # time_stamps = time_stamps_original[0].detach().clone()[mask_indices].to(device)
        # time_stamps = time_stamps.reshape(1, -1, 1).repeat(timeSeries_noisy.size(0), 1, 1)
        # timeSeries_noisy = torch.cat((timeSeries_noisy.to(device), time_stamps), dim=-1).float()

        out = model(timeSeries_noisy_original.unsqueeze(-1), mask)
        # pz0_mean = pz0_logvar = None
        out, idx = out

        # Use loss weights from config
        loss, l1_loss, gradient_loss = model.calculate_loss(
            out,
            groundTruth,
            time_interval=torch.linspace(0, 1, steps=cfg.number_x_values),
            weight_l1=getattr(cfg, 'weight_l1', 1.0),
            weight_grad=getattr(cfg, 'weight_grad', 0.5),
            weight_smooth=getattr(cfg, 'weight_smooth', 0.1),
            idx=idx,
        )
        loss.backward()
        if getattr(cfg, 'grad_clip', None) and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_meter.update(loss.item())
        epoch_loss += loss.item()
        num_batches += 1

        if not cfg.disable_tqdm:
            iterable.set_postfix({'loss': f'{loss_meter.avg:.4f}',
                                  'L1': f'{l1_loss.item():.4f}',
                                  'Grad': f'{gradient_loss.item():.4f}',
                                  'batch': f'{batch_idx}/{total}'}, refresh=False)
        if batch_idx % cfg.log_step == 0 or batch_idx == total:
            log.info('Training: Iter: %d, batch: %d/%d, running loss: %.4f',
                     epoch, batch_idx, total, loss_meter.avg)

    return epoch_loss / max(num_batches, 1)


def run_validation(model, val_loader, device, epoch, cfg, log, save_visuals=True):
    """
    Run validation and return metrics for early stopping.

    Returns:
        dict: Dictionary containing MAE, RMSE, and other metrics
    """
    model.eval()
    metrics = {'mae': float('inf'), 'rmse': float('inf')}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            groundTruth_cpu = batch["groundTruth"]
            timeSeries_noisy_original = batch["noisy_TimeSeries"]
            mask = batch["mask"]
            time_stamps_original = batch["time_stamps"]

            div_term_cpu = batch["div_term"]
            min_value_cpu = batch["min_value"]

            mask_indices = torch.where(mask[0] == True)[0]
            timeSeries_noisy = timeSeries_noisy_original[:, mask_indices].unsqueeze(-1)
            time_stamps = time_stamps_original[0].detach().clone()[mask_indices]
            time_stamps = time_stamps.reshape(1, -1, 1).repeat(timeSeries_noisy.size(0), 1, 1).to(device)
            timeSeries_noisy = torch.cat((timeSeries_noisy.to(device), time_stamps), dim=-1).float()

            pred_x = model(timeSeries_noisy, time_stamps_original[0])[0]
            div_term = div_term_cpu.unsqueeze(-1).unsqueeze(-1).to(device)
            min_value = min_value_cpu.unsqueeze(-1).unsqueeze(-1).to(device)
            groundTruth = groundTruth_cpu.unsqueeze(-1).to(device)

            # Inverse transform from [-1, 1] back to original scale
            pred_x = ((pred_x * 0.5) + 0.5) * div_term + min_value
            groundTruth = ((groundTruth * 0.5) + 0.5) * div_term + min_value

            mae = torch.abs(pred_x - groundTruth).mean()
            rmse = torch.sqrt(((pred_x - groundTruth) ** 2).mean())

            # Additional metrics
            mape = torch.mean(torch.abs((pred_x - groundTruth) / (groundTruth + 1e-8))) * 100

            metrics = {'mae': mae.item(), 'rmse': rmse.item(), 'mape': mape.item()}
            log.info('Validation: Epoch: %d, MAE: %.4f, RMSE: %.4f, MAPE: %.2f%%',
                     epoch, mae.item(), rmse.item(), mape.item())

            if save_visuals:
                # Denormalize time for visualization and exports (model still uses [0, 1])
                time_scale = float(cfg.number_x_values - 1) if cfg.number_x_values > 1 else 1.0
                time_stamps_plot = time_stamps_original[0] * time_scale

                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                ax.plot(time_stamps_plot,
                        ((timeSeries_noisy_original[0] * 0.5 + 0.5) * div_term_cpu[0]) + min_value_cpu[0],
                        'o', markersize=2, alpha=0.5, label="Noisy Input")
                ax.plot(time_stamps_plot,
                        groundTruth[0].detach().cpu().squeeze(-1),
                        linewidth=2, label="Ground Truth")
                ax.plot(time_stamps_plot,
                        pred_x[0].detach().cpu().squeeze(-1),
                        linewidth=2, linestyle='--', label="Prediction")
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'Epoch {epoch} | MAE: {mae.item():.4f} | RMSE: {rmse.item():.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                save_path = os.path.join(cfg.val_dir_pictures, f'vis_{epoch}.svg')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                loss_fig, loss_ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
                error_samples = np.arange(len(model.total_loss))
                loss_ax[0].plot(error_samples, model.total_loss, label="Total Loss", alpha=0.7)
                loss_ax[0].set_ylabel('Total Loss')
                loss_ax[1].plot(error_samples, model.L1_loss, label="L1 Loss", color='orange', alpha=0.7)
                loss_ax[1].set_ylabel('L1 Loss')
                loss_ax[2].plot(error_samples, model.gradient_loss, label="Gradient Loss", color='green', alpha=0.7)
                loss_ax[2].set_ylabel('Gradient Loss')
                loss_ax[2].set_xlabel('Training Steps')
                for ax in loss_ax:
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                loss_path = os.path.join(cfg.val_dir_pictures, f'loss_{epoch}.svg')
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                plt.close(loss_fig)

                loss_data_path = os.path.join(cfg.val_dir_data, f'loss_{epoch}.pkl')
                torch.save({
                    'total_loss': model.total_loss,
                    'l1_loss': model.L1_loss,
                    'gradient_loss': model.gradient_loss,
                    'smooth_loss': model.smooth_loss,
                }, loss_data_path)

                save_path = os.path.join(cfg.val_dir_data, f'pred_{epoch}.pkl')
                torch.save({
                    'pred': pred_x,
                    'target': groundTruth,
                    'samp': timeSeries_noisy,
                    'time_plot': time_stamps_plot,
                    'metrics': metrics,
                }, save_path)

            break

    return metrics


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=100, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True  # First call, save model

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved, save model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Not improved


@hydra.main(config_path="config", config_name="training", version_base=None)
def main(cfg: DictConfig):
    device, log = setup_environment(cfg)
    train_dataloader, val_dataloader = get_ds_timeSeries(cfg)
    model, optimizer, scheduler, start_itr = build_model_and_optimizer(cfg, device, log)
    loss_meter = RunningAverageMeter()

    # Initialize early stopping
    early_stopping = None
    if getattr(cfg, 'early_stopping', False):
        patience = getattr(cfg, 'patience', 200)
        min_delta = getattr(cfg, 'min_delta', 0.0001)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        log.info(f'Early stopping enabled with patience={patience}, min_delta={min_delta}')

    val_freq = max(1, cfg.val_plot_freq)
    best_val_loss = float('inf')
    warmup_epochs = getattr(cfg, 'warmup_epochs', 100)

    for itr in range(start_itr + 1, cfg.niters + 1):
        # Warmup learning rate
        if scheduler is not None and itr <= warmup_epochs:
            warmup_lr = cfg.lr * (itr / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        epoch_loss = train_one_epoch(model, optimizer, train_dataloader, device, itr, loss_meter, cfg, log)

        # Step scheduler after warmup
        if scheduler is not None and itr > warmup_epochs:
            scheduler_type = getattr(cfg, 'scheduler_type', 'cosine')
            if scheduler_type == 'plateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if itr % val_freq == 0:
            log.info(f'Epoch {itr}: LR = {current_lr:.6f}')

        save_checkpoint(model, optimizer, scheduler, itr, cfg, log)
        save_visuals = (itr % val_freq == 0)
        metrics = run_validation(model, val_dataloader, device, itr, cfg, log, save_visuals=save_visuals)

        # Early stopping and best model saving
        if early_stopping is not None:
            is_best = early_stopping(metrics['mae'])
            if is_best:
                save_checkpoint(model, optimizer, scheduler, itr, cfg, log, best=True)
                best_val_loss = metrics['mae']
                log.info(f'New best model saved! MAE: {best_val_loss:.4f}')

            if early_stopping.early_stop:
                log.info(f'Early stopping triggered at epoch {itr}. Best MAE: {best_val_loss:.4f}')
                break
        elif metrics['mae'] < best_val_loss:
            best_val_loss = metrics['mae']
            save_checkpoint(model, optimizer, scheduler, itr, cfg, log, best=True)

    log.info(f'Training completed. Best validation MAE: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
