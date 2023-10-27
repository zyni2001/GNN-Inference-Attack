import os
# exp related
RAW_DATA_PATH = 'temp_data/raw_data/'
PROCESSED_DATA_PATH = 'temp_data/processed_data/'
TARGET_MODEL_PATH = 'temp_data/target_model/'
SHADOW_MODEL_PATH = 'temp_data/shadow_model/'
GAE_MODEL_PATH = 'temp_data/gae_model/'
SPLIT_PATH = 'temp_data/split/'
ATTACK_DATA_PATH = 'temp_data/attack_data/'
ATTACK_MODEL_PATH = 'temp_data/attack_model/'
DEFENSE_DATA_PATH = 'temp_data/defense_data/'
LOG_PATH = 'temp_data/log/txt/'
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
