import argparse
import os
from pathlib import Path

import torch

from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType, LossType
from src.diffusion.respace import SpacedDiffusion, space_timesteps
from src.model.unet import UNetModel
from src.trainer.trainer import Trainer
from src.utils.utils import load_config_from_yaml, set_seed_everywhere


OBJECTIVE = {
    'PREVIOUS_X': ModelMeanType.PREVIOUS_X,
    'START_X': ModelMeanType.START_X,
    'EPSILON': ModelMeanType.EPSILON,
    'VELOCITY': ModelMeanType.VELOCITY,
}
MODEL_VAR_TYPE = {
    'LEARNED': ModelVarType.LEARNED,
    'FIXED_SMALL': ModelVarType.FIXED_SMALL,
    'FIXED_LARGE': ModelVarType.FIXED_LARGE,
    'LEARNED_RANGE': ModelVarType.LEARNED_RANGE,
}
LOSS_TYPE = {'l1': LossType.MAE, 'l2': LossType.MSE}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / 'src' / 'config' / 'pasta_mri2pet_parkinson_mid_eval.yaml'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default=str(DEFAULT_CONFIG),
)
cli_args, _ = parser.parse_known_args()


def _resolve_existing_path(value, fallback: Path) -> str:
    if value:
        candidate = Path(str(value))
        if candidate.exists():
            return str(candidate)
    return str(fallback)


def _resolve_existing_dir(value, fallback: Path) -> str:
    resolved = _resolve_existing_path(value, fallback)
    return str(Path(resolved)) + os.sep


def main():
    set_seed_everywhere(666)
    args = load_config_from_yaml(cli_args.config)

    local_data_dir = PROJECT_ROOT / 'data'
    local_results_dir = PROJECT_ROOT / 'results_parkinson'

    args.data_dir = _resolve_existing_dir(getattr(args, 'data_dir', None), local_data_dir)
    args.results_folder = _resolve_existing_path(getattr(args, 'results_folder', None), local_results_dir)

    model = UNetModel(
        image_size=args.image_size,
        in_channels=args.model_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_model,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        cond_emb_channels=args.cond_emb_channels,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_model,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
    )

    encoder = UNetModel(
        image_size=args.image_size,
        in_channels=args.encoder_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_encoder,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_encoder,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
    )

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(args.timesteps, args.timestep_respacing),
        model=model,
        encoder=encoder,
        beta_schedule=args.beta_schedule,
        timesteps=args.timesteps,
        model_mean_type=OBJECTIVE[args.objective],
        model_var_type=MODEL_VAR_TYPE[args.model_var_type],
        loss_type=LOSS_TYPE[args.loss_type],
        gen_type=args.gen_type,
        use_fp16=False,
        condition=args.condition,
        reconstructed_loss=args.reconstructed_loss,
        recon_weight=args.recon_weight,
        rescale_intensity=args.rescale_intensity,
    )

    trainer = Trainer(
        diffusion,
        folder=args.data_dir,
        input_slice_channel=args.input_slice_channel,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        fp16=args.fp16,
        calculate_fid=args.calculate_fid,
        dataset=args.dataset,
        image_direction=args.image_direction,
        num_slices=args.num_slices,
        tabular_cond=args.tabular_cond,
        results_folder=args.results_folder,
        resume=None,
        pretrain=None,
        test_batch_size=args.test_batch_size,
        eval_mode=True,
        eval_dataset=args.eval_dataset,
        eval_resolution=args.eval_resolution,
        model_cycling=args.model_cycling,
        ROI_mask=args.ROI_mask,
        dx_labels=args.dx_labels,
    )

    checkpoint_path = Path(args.results_folder) / 'best_val_model.pt'
    if not checkpoint_path.exists():
        fallback_checkpoint = local_results_dir / 'best_val_model.pt'
        if fallback_checkpoint.exists():
            checkpoint_path = fallback_checkpoint
        else:
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt['model'], strict=False)
    trainer.ema.load_state_dict(ckpt['ema'], strict=False)

    diffusion_model = trainer.ema.ema_model
    unet = diffusion_model.model
    encoder = diffusion_model.encoder
    unet.eval()
    print(f'Checkpoint cargado: {checkpoint_path}')

    output_dir = Path(args.results_folder) / 'adagn_features'
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {i: label for i, label in enumerate(args.dx_labels)}
    device = trainer.device

    print(f'\nExtrayendo AdaGN -> {output_dir}\n')

    for test_data in trainer.dl_test:
        mri_slices = test_data[0]
        spect_slices = test_data[1]
        dx_label = test_data[2]
        mri_uid = test_data[-1]
        tabular_data = test_data[3].to(device) if trainer.tabular_cond else None
        
        
        print("=" * 60)
        print("Pacientes:", mri_uid)
        print("len(mri_slices) =", len(mri_slices))
        print("shape ventana 0 =", mri_slices[0].shape)
        print("=" * 60)
  
        # Verificar si todos los pacientes del batch ya existen
        all_exist = True

        for uid in mri_uid:
            output_file = output_dir / f"{str(uid)}.pt"

            if not output_file.exists():
                all_exist = False
                break

        if all_exist:
            print(f"Saltando batch completo: {list(mri_uid)}")
            continue


        slice_features = []

        for slice_num in range(len(mri_slices)):
            mri_slice = mri_slices[slice_num].to(device)
            spect_slice = spect_slices[slice_num].to(device)

            with torch.no_grad():
                # Forzamos evaluación explícita
                _, cond_features = encoder(mri_slice, return_features=True, tab_cond=tabular_data)
                timesteps = torch.zeros(mri_slice.shape[0], dtype=torch.long, device=device)
                _ = unet(x=spect_slice, timesteps=timesteps, cond=cond_features, tab_cond=tabular_data)

            # 1. Extraer con .detach() para evitar fugas de memoria VRAM
            feat1 = unet.middle_block[0].last_adagn.detach()
            feat2 = unet.middle_block[2].last_adagn.detach()
            
            # 2. Asegurar reducción espacial correcta basándonos en dimensiones
            # Si tiene 4D (B, C, H, W) promedia las últimas dos
            if len(feat1.shape) == 4:
                feat1 = feat1.mean(dim=(-1, -2))
            if len(feat2.shape) == 4:
                feat2 = feat2.mean(dim=(-1, -2))

            feat = torch.cat([feat1, feat2], dim=1)
            
            slice_features.append(feat)

        # patient_feature tendrá dimensiones: [Batch_Size, Channels]
        patient_feature = torch.stack(slice_features, dim=0).mean(dim=0)

        for i in range(len(mri_uid)):
            uid = str(mri_uid[i])
            output_file = output_dir / f"{uid}.pt"

            if output_file.exists():
                print(f"Saltando paciente {uid}")
                continue

            label = int(dx_label[i].item())
            label_str = label_map.get(label, str(label))
            
            # Extraemos el vector limpio del paciente 'i' del lote actual
            individual_feature = patient_feature[i].cpu().float()

            torch.save(
                {
                    'uid': uid,
                    'label': label,
                    'label_str': label_str,
                    'adagn_feature': individual_feature,
                },
                output_file,
            )

            print(f'  {uid}  ({label_str})  ->  {individual_feature.shape}')
            
        # Limpieza explícita del estado al terminar el batch para obligar al modelo a recalcular
        if hasattr(unet.middle_block[0], 'last_adagn'):
            del unet.middle_block[0].last_adagn

        if hasattr(unet.middle_block[2], 'last_adagn'):
            del unet.middle_block[2].last_adagn

    n_saved = len(list(output_dir.glob('*.pt')))
    print(f'\nListo. {n_saved} pacientes guardados en {output_dir}')


if __name__ == '__main__':
    main()
