#!/bin/bash

n_epochs=100
warmup_epochs=20
loader="DAM-VP"
dataset="cifar100"
batch_size=64
scheduler="cosine"
optimizer="AdamW"
p_len=20
wd=0.0005
cs=("0.003")
trainer="ours"
mixup="mixup"
transform="default"
net=19
prompt_patch=16
hid_dim=16

as=("7")
bs=("9")
ds=("3")
deep_layer=12
type="ours9"
cutmix_alpha="1.0"
bn="none"
prompt_patch_2=11
prompt_patch_22=25
simam=False


for k1 in "${as[@]}"
do
    for k2 in "${bs[@]}"
    do
        for lr in "${cs[@]}"
        do
            for k3 in "${ds[@]}"
            do
                python main.py --info="InstanceVPD-${net}-n=${prompt_patch}-h=${hid_dim}-n2=${prompt_patch_2}-n22=${prompt_patch_22}-w=2-lr=${lr}-bs=${batch_size}-dl=${deep_layer}-mixup=${mixup}-${cutmix_alpha}-drop-${k1}-${k2}-${k3}-p_len=${p_len}" \
                    --model=InstanceVPD  \
                    --output_path="./Output/${dataset}/ViT/imagenet22k/InsVP" \
                    --n_epochs=$n_epochs --meta_net=$net --hid_dim=$hid_dim --prompt_patch=$prompt_patch \
                    --pretrained=imagenet22k --prompts_2_weight=2 \
                    --batch_size=$batch_size \
                    --prompt_patch_2=$prompt_patch_2 --prompt_patch_22=$prompt_patch_22 \
                    --dataset=$dataset \
                    --lr=$lr --scheduler=$scheduler \
                    --weight_decay=$wd --optimizer=$optimizer \
                    --loader=$loader \
                    --transform=$transform \
                    --deep_prompt_type=${type} \
                    --deep_layer=${deep_layer} \
                    --mixup=${mixup} \
                    --cutmix_alpha=${cutmix_alpha} \
                    --TP_kernel_1=${k1} \
                    --TP_kernel_2=${k2} \
                    --TP_kernel_3=${k3} \
                    --p_len=${p_len} \
                    --p_len_vpt=${p_len} \
                    --trainer=${trainer} \
                    --simam=${simam} \
                    --warmup_epochs=$warmup_epochs \
                    --base_dir="your_data_path"
            done
        done
    done
done


# chmod +x ./sh/cifar100.sh
# bash ./sh/cifar100.sh
