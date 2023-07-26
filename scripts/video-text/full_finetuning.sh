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


lr=$2
epoch=$3
seed=$4
sh=full_finetuning_lr${lr}_epoch${epoch}_seed${seed}
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
    --unfreeze_language_model \
    --seed ${seed}
