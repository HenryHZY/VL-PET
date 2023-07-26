task=multitask

model="t5"
folder_prefix="VLT5"
backbone="t5-base"
batch_size=300

echo $folder_prefix
echo $backbone

feature=RN101

lr=$6
sh=T5_bs300_Encoder_MultiheadDownAdapter_dim$2_head$3_MultiheadUpZeroInit_GatingSmall_xycatLN_GatingScaling$4_GatingUpZeroInit_Decoder_VPAdapter_dim$5_VPAUpZeroInit_lr$6_seed$7
name=${sh}_${feature}__bs${batch_size}_image224_lr${lr}
output=snap/${folder_prefix}_${task}/$name

TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$1 \
    src/${task}.py \
    --distributed --multiGPU \
    --optim adamw \
    --warmup_ratio 0.1 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 20 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output \
    --num_beams 5 \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --reduction_factor 8 \
    --use_tasks_prompts \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} --n_boxes 36 --downsample \
    --image_size "(224,224)" \
    --run_name $name \
    --use_adapter \
    --use_single_adapter \
    --no_encoder_adapter \
    --use_adapter_down_dim \
    --use_encoder_adapter_down_multihead \
    --adapter_down_dim $2 \
    --encoder_adapter_multihead_num_head $3 \
    --use_encoder_multihead_up_zero_init \
    --use_encoder_adapter_gating_small_xy_cat \
    --use_encoder_gating_scaling \
    --encoder_gating_scaling_factor $4 \
    --use_encoder_gating_small_up_zero_init \
    --unfreeze_encoder_layer_norms \
    --no_decoder_adapter \
    --use_decoder_enc_attn_value_parallel_adapter_down_dim \
    --decoder_enc_attn_value_parallel_adapter_down_dim $5 \
    --use_decoder_enc_vpa_up_zero_init \
    --seed $7
