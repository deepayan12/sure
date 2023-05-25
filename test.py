import argparse
import pandas as pd
import numpy as np
from functools import partial

import runner
import sim_with_classes_helper

# CUDA_VISIBLE_DEVICES=1 /usr/bin/python3 -u test.py --dataset example --dropout_p 0.5 --epochs 2000 --num_procs 5 --num_seeds 5

def parse_args():
  parser = argparse.ArgumentParser(description='Fairness on real datasets')
  parser.add_argument('--dataset', type=str, default='example', help='npz file with data')
  parser.add_argument('--num_procs', type=int, default=15, help='Number of processes')
  parser.add_argument('--num_seeds', type=int, default=30, help='Number of repetitions')
  parser.add_argument('--skip_epochs', type=int, default=10, help='Number of initial epochs before SURE starts looking for high-risk boxes')
  parser.add_argument('--cluster_epochs', type=int, default=30, help='Number of epochs after each clustering step')
  parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
  parser.add_argument('--num_pts_in_bin', type=int, default=30, help='Min number of points needed in cluster bin')
  parser.add_argument('--dropout_p', type=float, default=0, help='Dropout probability')
  
  args = parser.parse_args()
  print(args)
  return args


def do_all():
  pd.set_option('display.max_rows', 100, 'display.width', 1000, 'display.max_columns', None, 'display.max_colwidth', None)
  args = parse_args()

  all_args = []
  for seed in np.arange(args.num_seeds):
    F = np.load(f'{args.dataset}.npz', allow_pickle=True)
    X_train, X_train_sensitive, y_train, X_test, X_test_sensitive, y_test, sensitive_cols = \
        F['X_train'], F['X_train_sensitive'], F['y_train'], F['X_test'], F['X_test_sensitive'], F['y_test'], F['sensitive_cols'].item()


    training_data = runner.MyDataset(X_train, X_train_sensitive, y_train)
    testing_data = runner.MyDataset(X_test, X_test_sensitive, y_test)

    verbose = -3
    arg_func = partial(runner.do_all, training_data=training_data, testing_data=testing_data, sensitive_cols=sensitive_cols, batch_size=200, num_pts_in_bin=args.num_pts_in_bin, skip_epochs=args.skip_epochs, cluster_epochs=args.cluster_epochs, epochs=args.epochs, verbose=verbose, dropout_p=args.dropout_p, train_error_check_epochs=[50, 100, 150, 200]+list(np.arange(300, args.epochs+1, 500)))
    all_args.append((seed, arg_func))

  all_res, all_res_full, attr_series = sim_with_classes_helper.create_and_run_args(list_seeds_and_arg_funcs=all_args, num_procs=args.num_procs)


if __name__ == '__main__':
  do_all()
