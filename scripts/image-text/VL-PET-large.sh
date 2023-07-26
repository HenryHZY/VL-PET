task=multitask
proxy_on="export http_proxy=http://huziyuan:qwer520--@10.1.8.50:33128/ ; export https_proxy=http://huziyuan:qwer520--@10.1.8.50:33128/ ; export HTTP_PROXY=http://huziyuan:qwer520--@10.1.8.50:33128/ ; export HTTPS_PROXY=http://huziyuan:qwer520--@10.1.8.50:33128/"
eval ${proxy_on}

model="bart"

echo $model

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=300
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=500
fi

echo $folder_prefix
echo $backbone

feature=RN101

lr=$6
sh=Encoder_MultiheadDownAdapter_dim$2_head$3_GatingLowRankLN_dim$4_Decoder_VPAdapter_dim$5_lr$6_seed$7
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
    --use_encoder_adapter_gating_large_x_lowrank \
    --adapter_gating_down_dim $4 \
    --unfreeze_encoder_layer_norms \
    --no_decoder_adapter \
    --use_decoder_enc_attn_value_parallel_adapter_down_dim \
    --decoder_enc_attn_value_parallel_adapter_down_dim $5 \
    --seed $7
