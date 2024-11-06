import pandas as pd
from os.path import join as joindir
from os import makedirs as mkdir

def pytest(args):
    RESULT_DIR = joindir('./result', '.'.join(__file__.split('.')[:-1]))
    mkdir(RESULT_DIR, exist_ok=True)
    record_dfs = []
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_record = pd.DataFrame(ppo(args))
        reward_record['#parallel_run'] = i
        record_dfs.append(reward_record)
    record_dfs = pd.concat(record_dfs, axis=0)
    record_dfs.to_csv(joindir(RESULT_DIR, args.algorihtm + '-record-{}.csv'.format(args.env_name)))





















