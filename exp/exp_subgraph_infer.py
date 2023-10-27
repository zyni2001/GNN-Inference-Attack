import logging

import numpy as np

from exp.exp import Exp
from lib_subgraph_infer.attack_subgraph_infer import AttackSubgraphInfer
import concurrent.futures

class ExpSubgraphInfer(Exp):
    def __init__(self, args):
        super(ExpSubgraphInfer, self).__init__(args)
        self.logger = logging.getLogger('exp_subgraph_infer_2')
        # print args value
        for key, value in self.args.items():
            self.logger.info('%s: %s' % (key, value))
            
        self.acc_run = []
        self.auc_run = []
        self.acc = {}
        self.auc = {}

        self.launch_attack()
        self.cal_stat()

    def launch_attack(self):
        self.logger.info('launching attack')

        for run in range(self.args['num_runs']):
            # self.train_target_model()
            # load target model and its corresponding parameters
            # if not self.args['is_train_target_model']:
            #     paras = self.data_store.load_target_model(self.target_model)
            # else:
            #     paras = self.target_model.paras
            paras = {}
            paras['embedding_dim'] = 192

            # load shadow model and its corresponding parameters
            # if self.args['is_use_shadow_model']:
            #     if not self.args['is_train_shadow_model']:
            #         paras = self.data_store.load_shadow_model(self.shadow_model)
            #     else:
            #         paras = self.shadow_model.paras

            if self.args['is_use_shadow_model']:
                attack = AttackSubgraphInfer(self.target_model.model, self.shadow_model.model, paras['embedding_dim'], self.dataset.num_classes, self.args)
            else:
                attack = AttackSubgraphInfer(self.target_model.model, self.target_model.model, paras['embedding_dim'], self.dataset.num_classes, self.args)

        
            self.logger.info('%s run' % (run,))
            # generate attack training data
            if self.args['is_gen_attack_data']:
                # attack_train_dataset = self.dataset[list(self.attack_train_indices)]
                # attack_test_dataset = self.dataset[list(self.attack_test_indices)]

                
                attack.determine_subsample_cls(self.args['train_sample_method'])
                # attack.generate_train_data(self.attack_train_dataset, self.attack_test_dataset)
                attack.generate_train_data(self.attack_train_dataset, self.sub_train_neg_dataset)

                attack.determine_subsample_cls(self.args['test_sample_method'])
                # attack.generate_test_data(self.attack_test_dataset, self.attack_train_dataset)
                attack.generate_test_data(self.attack_test_dataset, self.sub_test_neg_dataset)

                self.data_store.save_subgraph_infer_2_data(attack)

                # def generate_train_data_task():
                #     attack.determine_subsample_cls(self.args['train_sample_method'])
                #     # attack.generate_train_data(self.attack_train_dataset, self.attack_test_dataset)
                #     attack.generate_train_data(self.attack_train_dataset, self.sub_train_neg_dataset)

                # def generate_test_data_task():
                #     attack.determine_subsample_cls(self.args['test_sample_method'])
                #     # attack.generate_test_data(self.attack_test_dataset, self.attack_train_dataset)
                #     attack.generate_test_data(self.attack_test_dataset, self.sub_test_neg_dataset)

                # # Using ThreadPoolExecutor to run the tasks in parallel
                # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                #     # Execute the two tasks concurrently
                #     future1 = executor.submit(generate_train_data_task)
                #     future2 = executor.submit(generate_test_data_task)

                #     # Optional: If you want to wait for both tasks to complete before moving on
                #     concurrent.futures.wait([future1, future2])

                # self.data_store.save_subgraph_infer_2_data(attack)

            else:
                if self.args['is_use_our_embedding']:
                    self.data_store.load_subgraph_infer_2_data_our(attack)
                    attack.generate_train_data_our(self.attack_train_dataset, self.sub_train_neg_dataset, self.attack_train_indices, self.attack_test_indices, self.embed_list_train)
                    attack.generate_test_data_our(self.attack_test_dataset, self.sub_test_neg_dataset, self.attack_test_indices, self.attack_train_indices, self.embed_list_test)
                else:
                    self.data_store.load_subgraph_infer_2_data(attack)

            acc, auc = {}, {}
            # train and test attack model
            for feat_gen_method in self.args['feat_gen_method']:
                self.logger.info('feat_gen_method: %s' % (feat_gen_method,))
                attack.determine_feat_gen_fn(feat_gen_method)
                attack.generate_dataloader()

                attack.train_attack_model(self.dataset.num_features, feat_gen_method)
                self.data_store.save_subgraph_infer_2_model(attack)
                acc[feat_gen_method], auc[feat_gen_method] = attack.evaluate_attack_model()

            # print the acc and auc of each run
            for feat_gen_method in self.args['feat_gen_method']:
                self.logger.info('run %s, config: %s, attack acc: %s, attack auc %s' % (
                    run, feat_gen_method, acc[feat_gen_method], auc[feat_gen_method]))
            self.acc_run.append(acc)
            self.auc_run.append(auc)

    def cal_stat(self):
        self.logger.info('calculating statistics')

        for feat_gen_method in self.args['feat_gen_method']:
            acc_run_data = np.zeros(self.args['num_runs'])
            auc_run_data = np.zeros(self.args['num_runs'])

            for run in range(self.args['num_runs']):
                acc_run_data[run] = self.acc_run[run][feat_gen_method]
                auc_run_data[run] = self.auc_run[run][feat_gen_method]
            
            # self.acc[feat_gen_method] = [np.mean(acc_run_data), np.std(acc_run_data)]
            # self.auc[feat_gen_method] = [np.mean(auc_run_data), np.std(auc_run_data)]

            # self.logger.info('config: %s, attack acc: %s, attack auc %s' % (
                # feat_gen_method, self.acc[feat_gen_method], self.auc[feat_gen_method]))
            
            # print all the values of acc and auc
            self.logger.info('config: %s, attack acc: %s, attack auc %s' % (
                feat_gen_method, acc_run_data, auc_run_data))
            # print the maximum value of acc and auc
            self.logger.info('config: %s, max attack acc: %s, max attack auc %s' % (
                feat_gen_method, np.max(acc_run_data), np.max(auc_run_data)))
