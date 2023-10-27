import os
import sys
# exp related
# create temp_data dir according to the iteration number
EXP_NAME = 'temp_data_iteration_{}'.format(sys.argv[1])
# RAW_DATA_PATH = 'temp_data/raw_data/'
# PROCESSED_DATA_PATH = 'temp_data/processed_data/'
# TARGET_MODEL_PATH = 'temp_data/target_model/'
# SHADOW_MODEL_PATH = 'temp_data/shadow_model/'
# GAE_MODEL_PATH = 'temp_data/gae_model/'
# SPLIT_PATH = 'temp_data/split/'
# ATTACK_DATA_PATH = 'temp_data/attack_data/'
# ATTACK_MODEL_PATH = 'temp_data/attack_model/'
# DEFENSE_DATA_PATH = 'temp_data/defense_data/'
# LOG_PATH = 'temp_data/log/txt/'
# create directories according to the EXP_NAME
RAW_DATA_PATH = EXP_NAME + '/raw_data/'
PROCESSED_DATA_PATH = EXP_NAME + '/processed_data/'
TARGET_MODEL_PATH = EXP_NAME + '/target_model/'
SHADOW_MODEL_PATH = EXP_NAME + '/shadow_model/'
GAE_MODEL_PATH = EXP_NAME + '/gae_model/'
SPLIT_PATH = EXP_NAME + '/split/'
ATTACK_DATA_PATH = EXP_NAME + '/attack_data/'
ATTACK_MODEL_PATH = EXP_NAME + '/attack_model/'
DEFENSE_DATA_PATH = EXP_NAME + '/defense_data/'
LOG_PATH = EXP_NAME + '/log/txt/'
# create directories
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(TARGET_MODEL_PATH, exist_ok=True)
os.makedirs(SHADOW_MODEL_PATH, exist_ok=True)
os.makedirs(GAE_MODEL_PATH, exist_ok=True)
os.makedirs(SPLIT_PATH, exist_ok=True)
os.makedirs(ATTACK_DATA_PATH, exist_ok=True)
os.makedirs(ATTACK_MODEL_PATH, exist_ok=True)
os.makedirs(DEFENSE_DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
