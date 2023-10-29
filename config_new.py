# config.py
import os

class Config:
    def __init__(self, config): # DD_0.2_diff_pool_snow_ball 
        self.EXP_NAME = 'temp_data_{}'.format(config)
        self.RAW_DATA_PATH = self.EXP_NAME + '/raw_data/'
        self.PROCESSED_DATA_PATH = self.EXP_NAME + '/processed_data/'
        self.TARGET_MODEL_PATH = self.EXP_NAME + '/target_model/'
        self.SHADOW_MODEL_PATH = self.EXP_NAME + '/shadow_model/'
        self.GAE_MODEL_PATH = self.EXP_NAME + '/gae_model/'
        self.SPLIT_PATH = self.EXP_NAME + '/split/'
        self.ATTACK_DATA_PATH = self.EXP_NAME + '/attack_data/'
        self.ATTACK_MODEL_PATH = self.EXP_NAME + '/attack_model/'
        self.DEFENSE_DATA_PATH = self.EXP_NAME + '/defense_data/'
        self.LOG_PATH = self.EXP_NAME + '/log/txt/'
        # extract the pooling method from the config
        self.POOLING_METHOD = config.split('_')[-4]
        self.EMBEDDING_PATH = '/home/zhiyu/GNN-Embedding-Leaks-DD' + f'/DD_embedding_{self.POOLING_METHOD}_pool/'
        self.SKIP_INDICES_PATH = '/home/zhiyu/GNN-Embedding-Leaks-DD/skiped_indices/'
        # create directories
        os.makedirs(self.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(self.TARGET_MODEL_PATH, exist_ok=True)
        os.makedirs(self.SHADOW_MODEL_PATH, exist_ok=True)
        os.makedirs(self.GAE_MODEL_PATH, exist_ok=True)
        os.makedirs(self.SPLIT_PATH, exist_ok=True)
        os.makedirs(self.ATTACK_DATA_PATH, exist_ok=True)
        os.makedirs(self.ATTACK_MODEL_PATH, exist_ok=True)
        os.makedirs(self.DEFENSE_DATA_PATH, exist_ok=True)
        os.makedirs(self.LOG_PATH, exist_ok=True)
        os.makedirs(self.EMBEDDING_PATH, exist_ok=True)

