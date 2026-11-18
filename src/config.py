import os
from os.path import join, dirname

SEED = 42

ROOT_DIR = dirname(dirname(os.path.abspath(__file__))) #twice to get the workspace folder

TUNING_PATH = join(ROOT_DIR, 'checkpoint_tuning') + "/"

DATA_DIR = join(ROOT_DIR, 'data')
CACHE_DIR = join(ROOT_DIR, 'cache')


NETWORK_DATA_DIR = join(ROOT_DIR, 'data/Network')
NETWORK_DATA_PATH = NETWORK_DATA_DIR + "/"

Electricity_DATA_DIR = join(ROOT_DIR, 'data/Electricity')
Electricity_DATA_PATH = Electricity_DATA_DIR + "/"

Road_DATA_DIR = join(ROOT_DIR, 'data/Road')
Road_DATA_PATH = Road_DATA_DIR + "/"


FIG_DIR = join(ROOT_DIR, 'figures')
FIG_PATH = FIG_DIR + "/"

FIG_IGNORE_DIR = join(ROOT_DIR, 'figures_ignore')
FIG_IGNORE_PATH = FIG_IGNORE_DIR + "/"

RESULTS_DIR = join(ROOT_DIR, 'results')
RESULTS_PATH = RESULTS_DIR + "/"

RESULT_PHASE0_DIR = join(RESULTS_DIR, 'phase0')


LOG_DIR = join(ROOT_DIR, 'log')

EXP1_RESULTS_PATH = RESULTS_PATH + 'Diagnose/'
EVALUATION_RESULTS_PATH = RESULTS_PATH + 'Evaluation/'


os.makedirs(DATA_DIR, exist_ok= True)
os.makedirs(CACHE_DIR, exist_ok= True)
os.makedirs(LOG_DIR, exist_ok= True)
os.makedirs(NETWORK_DATA_DIR, exist_ok= True)
os.makedirs(Electricity_DATA_DIR, exist_ok= True)
os.makedirs(Road_DATA_DIR, exist_ok= True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULT_PHASE0_DIR, exist_ok = True)
os.makedirs(EXP1_RESULTS_PATH, exist_ok=True)
os.makedirs(EVALUATION_RESULTS_PATH, exist_ok=True)
