#!/bin/bash
gpu_counter=0
task_counter=0
use_gpu=8
use_task=$((3*$use_gpu))
itrs=21
log_path="temp_data_DD_0.2_diff_pool_snow_ball/embed_log_correct_setting/"
mkdir -p ${log_path}
# rm -rf temp_data_DD_0.2_diff_pool_snow_ball/raw_data/500/feat/DD/
repeats=4
for ((repeat=0;repeat<$repeats;repeat++)); do
        for ((i=0;i<$itrs;i++)); do
                cmd="CUDA_VISIBLE_DEVICES=$gpu_counter python -u main.py --configure DD_0.2_diff_pool_snow_ball --is_vary False --dataset DD --shadow_dataset DD \
                --exp 'subgraph_infer' --cuda 0 --num_threads 1 --is_split False --is_use_feat True --is_train_target_model False \
                --is_use_shadow_model False --is_train_shadow_model False --is_upload False --num_runs 1 --target_model diff_pool \
                --shadow_model diff_pool --max_nodes 500 --target_ratio 0.4 --attack_train_ratio 0.3 --attack_test_ratio 0.3 --batch_size 32 \
                --num_epochs 100 --is_gen_attack_data False --train_sample_method snow_ball --test_sample_method snow_ball --sample_node_ratio 0.2 \
                --attack 'subgraph_infer_2' --is_use_our_embedding True \
                --embed_train_itr 0 --embed_test_itr $i \
                > ${log_path}/${i}_repeat_${repeat}.log 2>&1 &"

                echo $cmd
                eval $cmd

                # Increment the GPU counter
                ((gpu_counter++))
                # Increment the task counter
                ((task_counter++))
                # If GPU counter reaches 8, reset it to 0 (though for 5 tasks this is unnecessary, it's good to keep in case you expand tasks in future)
                if [ $gpu_counter -eq $use_gpu ]; then
                        gpu_counter=0
                fi
                # If task counter reaches 5, wait for all tasks to finish and reset it to 0
                if [ $task_counter -eq $use_task ]; then
                        wait
                        task_counter=0
                        gpu_counter=0
                fi
        done
done