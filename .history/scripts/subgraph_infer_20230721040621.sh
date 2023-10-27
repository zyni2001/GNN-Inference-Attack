#!/bin/bash
# run.sh
cuda=$1
# dataset=$2
datasets=(DD AIDS ENZYMES)
sample_node_ratios=(0.2 0.4 0.6 0.8)
target_models=(diff_pool mincut_pool mean_pool)
sample_methods=(random_walk snow_ball forest_fire)
for dataset_name in ${datasets[@]}
do
  for sample_node_ratio in ${sample_node_ratios[@]}
    do
      for target_model in ${target_models[@]}
        do
          for sample_method in ${sample_methods[@]}
            do
              printf $dataset_name train_sample_method ratio
              python main.py --num_epoch 1 --shadow_dataset $dataset_name --is_train_target_model True --exp 'subgraph_infer' --attack 'subgraph_infer_2' --dataset $dataset_name --target_model $target_model --sample_node_ratio $sample_node_ratio  --train_sample_method $sample_method --test_sample_method $sample_method --cuda $cuda 2>&1 | tee "./temp_data/log/$dataset_name.$target_model.$sample_node_ratio.log" &
            done
        done
    done
done
wait
