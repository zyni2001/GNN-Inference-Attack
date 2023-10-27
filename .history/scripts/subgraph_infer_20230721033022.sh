#!/bin/bash
# run.sh
cuda=$1
# dataset=$2
for dataset_name in DD # AIDS, ENZYMES, DD
dataset=$dataset_name
do
  for sample_node_ratio in 0.4 # 0.2 0.4 0.6 0.8
    do
      for target_model in diff_pool # diff_pool, mincut_pool, mean_pool
        do
          for sample_method in snow_ball # random_walk, snow_ball, forest_fire
            do
              printf $dataset_name train_sample_method ratio
              python main.py --exp 'subgraph_infer' --attack 'subgraph_infer_2' --dataset $dataset_name --target_model $target_model --sample_node_ratio $sample_node_ratio  --train_sample_method $sample_method --test_sample_method $sample_method --cuda $cuda 2>&1 | tee "./temp_data/log/$dataset.$target_model.$sample_node_ratio.log" &
            done
        done
    done
done
wait
