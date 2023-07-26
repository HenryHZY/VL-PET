task=multitask_video


model="bart"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=30
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=50
fi

echo $folder_prefix
echo $backbone

feature=ViT




lr=$5
epoch=$6
seed=$7
sh=gating_middle_ia3_add_Encoder_MultiheadDownAdapter_dim$2_head$3_GatingMiddle_ia3addLN_Decoder_VPAdapter_dim$4_lr$5_epoch$6_seed$7
name=${sh}_bs${batch_size}_image224_lr${lr}_epoch${epoch}
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
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output \
    --num_beams 5 \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_tasks_prompts \
    --tasks "tvqa,how2qa,tvc,yc2c" \
    --feature ${feature} --n_boxes 64 --downsample \
    --image_size "(224,224)" \
    --run_name $name \
    --lr ${lr} \
    --epochs ${epoch} \
    --reduction_factor 8 \
    --use_adapter \
    --use_single_adapter \
    --no_encoder_adapter \
    --use_adapter_down_dim \
    --use_encoder_adapter_down_multihead \
    --adapter_down_dim $2 \
    --encoder_adapter_multihead_num_head $3 \
    --use_encoder_adapter_gating_middle_ia3_add \
    --unfreeze_encoder_layer_norms \
    --no_decoder_adapter \
    --use_decoder_enc_attn_value_parallel_adapter_down_dim \
    --decoder_enc_attn_value_parallel_adapter_down_dim $4 \
    --seed ${seed}

