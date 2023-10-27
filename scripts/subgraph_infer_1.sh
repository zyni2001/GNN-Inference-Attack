#!/bin/bash
# run.sh


# datasets=(DD) #(AIDS ENZYMES)
# sample_node_ratios=(0.2 0.4 0.6 0.8)
# target_models=(diff_pool mincut_pool mean_pool)
# sample_methods=(random_walk snow_ball forest_fire)

### Just test one setting for debug
datasets=(DD)
sample_node_ratios=(0.2)
target_models=(mean_pool)
sample_methods=(random_walk)
max_nodes=$1

num_gpus=7
gpu_id=0
tasks_on_gpu=0
for dataset_name in ${datasets[@]}
do
  for sample_node_ratio in ${sample_node_ratios[@]}
    do
      for target_model in ${target_models[@]}
        do
          for sample_method in ${sample_methods[@]}
            do
              # config is $dataset_$sample_node_ratio_$target_model_$sample_method
              config=$dataset_name"_"$sample_node_ratio"_"$target_model"_"$sample_method"_shadow_true"
              mkdir -p ./temp_data_$config/log/
              # paper
              # python main.py --attack 'subgraph_infer_2' --dataset $dataset_name --target_model $target_model --sample_node_ratio $sample_node_ratio  --train_sample_method $sample_method --test_sample_method $sample_method --cuda $cuda 2>&1 | tee "./temp_data/log/$dataset.$target_model.$sample_node_ratio.log" &
              
              ### setting 1: best auc=0.555
              # cmd="python main.py --is_vary False --dataset $dataset_name --shadow_dataset $dataset_name --exp 'subgraph_infer' --cuda $gpu_id --num_threads 1 --is_split False --is_use_feat True --is_train_target_model False --is_use_shadow_model False --is_train_shadow_model False --is_upload False --num_runs 1 --target_model $target_model --shadow_model $target_model --max_nodes 1000 --target_ratio 0.4 --attack_train_ratio 0.3 --attack_test_ratio 0.3 --batch_size 32 --num_epochs 100 --is_gen_attack_data False --train_sample_method $sample_method --test_sample_method $sample_method --sample_node_ratio $sample_node_ratio --attack 'subgraph_infer_2' 2>&1 | tee "./temp_data/log/$dataset_name.$target_model.$sample_node_ratio$sample_method.log" &"
              
              ### setting 2: best auc=0.697
              cmd="python main.py --configure $config --is_vary False --dataset $dataset_name --shadow_dataset $dataset_name --exp 'subgraph_infer' --cuda $gpu_id --num_threads 1 --is_split False --is_use_feat True --is_train_target_model True --is_use_shadow_model True --is_train_shadow_model True --is_upload False --num_runs 1 --target_model $target_model --shadow_model $target_model --max_nodes $max_nodes --target_ratio 0.2 --shadow_ratio 0.2 --attack_train_ratio 0.3 --attack_test_ratio 0.3 --batch_size 32 --num_epochs 100 --is_gen_attack_data True --train_sample_method $sample_method --test_sample_method $sample_method --sample_node_ratio $sample_node_ratio --attack 'subgraph_infer_2' 2>&1 | tee "./temp_data_$config/log/$dataset_name.$target_model.$sample_node_ratio.$sample_method.max_nodes$max_nodes.log" &"
              # cmd="python main.py --configure $config --is_vary False --dataset $dataset_name --shadow_dataset $dataset_name --exp 'subgraph_infer' --cuda $gpu_id --num_threads 1 --is_split False --is_use_feat True --is_train_target_model False --is_use_shadow_model False --is_train_shadow_model False --is_upload False --num_runs 1 --target_model $target_model --shadow_model $target_model --max_nodes $max_nodes --target_ratio 0.4 --attack_train_ratio 0.3 --attack_test_ratio 0.3 --batch_size 32 --num_epochs 300 --is_gen_attack_data False --train_sample_method $sample_method --test_sample_method $sample_method --sample_node_ratio $sample_node_ratio --attack 'subgraph_infer_2' 2>&1 | tee "./temp_data_$config/log/$dataset_name.$target_model.$sample_node_ratio.$sample_method.max_nodes$max_nodes.log" &"
              
              ### setting 3: best auc=0.614
              # cmd="python main.py --is_vary False --dataset $dataset_name --shadow_dataset $dataset_name --exp 'subgraph_infer' --cuda $gpu_id --num_threads 1 --is_split False --is_use_feat False --is_train_target_model False --is_use_shadow_model False --is_train_shadow_model False --is_upload False --num_runs 1 --target_model $target_model --shadow_model $target_model --max_nodes 1000 --target_ratio 0.2 --attack_train_ratio 0.4 --attack_test_ratio 0.4 --batch_size 32 --num_epochs 100 --is_gen_attack_data False --train_sample_method $sample_method --test_sample_method $sample_method --sample_node_ratio $sample_node_ratio --attack 'subgraph_infer_2' 2>&1 | tee "./temp_data/log/$dataset_name.$target_model.$sample_node_ratio$sample_method.log" &"
              
              echo $cmd              
              # eval $cmd

              tasks_on_gpu=$((tasks_on_gpu+1))
              if [ $tasks_on_gpu -eq 3 ]; then
                  tasks_on_gpu=0
                  gpu_id=$((gpu_id+1))
              fi
              if [ $gpu_id -gt $num_gpus ]; then
                  gpu_id=0
                  wait
              fi
            done
        done
    done
done
wait
