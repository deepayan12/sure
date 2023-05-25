from pandas import Series, DataFrame
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import time

# Increase num_repeats if you want to repeat each setting multiple times and take the median result
def run_one_arg(arg_tuple, num_repeats=1):
  seed, out_q, arg_func = arg_tuple
  print(seed, end=' ')
  all_res = []
  label, min_wt_factor, cluster_loss_multiplier = 'SURE', 1.0, 1.0
  for s in range(num_repeats):
    start_time = time.time()
    this_res_train, this_res_test, attr_series = arg_func(seed=seed+s, min_wt_factor=min_wt_factor, cluster_loss_multiplier=cluster_loss_multiplier)
    this_res_train = this_res_train.reset_index()
    this_res_train['Train/Test'] = 'Train'
    this_res_test = this_res_test.reset_index()
    this_res_test['Train/Test'] = 'Test'
    this_res = pd.concat([this_res_train, this_res_test], ignore_index=True)
    this_res['Method'] = label
    this_res['seed'] = seed
    this_res['repetition'] = s
    all_res.append(this_res)

  all_res_full = pd.concat(all_res, ignore_index=True)
  all_res = all_res_full.groupby(['index', 'Train/Test', 'epoch', 'Method', 'seed']).agg(lambda s: s.median()).drop('repetition', axis=1).reset_index()
  if out_q is not None:
    out_q.put((all_res, all_res_full, attr_series))
  else:
    return all_res, all_res_full, attr_series

def run_several_args(list_args):
  for arg_tuple in list_args:
    run_one_arg(arg_tuple)


def create_and_run_args(list_seeds_and_arg_funcs, num_procs):
  out_q = mp.Queue() if num_procs>1 else None
  all_args = [(x[0], out_q, x[1]) for x in list_seeds_and_arg_funcs]
  if num_procs == 1:
    results = map(run_one_arg, all_args)
    all_res, all_res_full, all_attr_series = list(zip(*results))
    attr_series = all_attr_series[0]
  else:
    num_args_each_proc = int(np.ceil(len(all_args) / num_procs))
    all_res = []
    all_res_full = []
    procs = []
    for i in range(num_procs):
      start_idx = i * num_args_each_proc
      end_idx = min(start_idx + num_args_each_proc, len(all_args))
      p = mp.Process(target=run_several_args, args=(all_args[start_idx:end_idx],))
      p.start()
      procs.append(p)
    for i in range(len(all_args)):
      this_res, this_res_full, attr_series = out_q.get()
      all_res.append(this_res)
      all_res_full.append(this_res_full)
    for p in procs:
      p.join()

  
  all_res = pd.concat(all_res, ignore_index=True)
  all_res = all_res.groupby(['index', 'Train/Test', 'epoch', 'Method', 'seed']).sum().unstack('seed').swaplevel(0,1,axis=1).sort_index(axis=1)
  all_res_full = pd.concat(all_res_full, ignore_index=True)
  all_res_full = all_res_full.groupby(['index', 'Train/Test', 'epoch', 'Method', 'seed', 'repetition']).sum().unstack(['seed', 'repetition']).swaplevel(0,1,axis=1).sort_index(axis=1)

  print()
  analyze_all_res(all_res, attr_series)
  return all_res, all_res_full, attr_series

def analyze_all_res(all_res, attr_series):
  all_z = all_res.loc['accuracy'].stack('seed').T.rename(attr_series).rename_axis(index='sensitive').apply(lambda s: s.groupby('sensitive').min()).sort_index(axis=0, level=['Train/Test', 'epoch', 'seed'])
  for sensitive in all_z.index.values:
    z = all_z.loc[sensitive].unstack(['Train/Test', 'epoch', 'seed'])
    print(f'==== Utility = worst-accuracy among all groups based on {sensitive} ({", ".join(attr_series.index.values)})')
    z2 = z.stack('seed').unstack('Method').sort_index(axis=1)
    print(z2)
    print(z2.describe().round(2))
