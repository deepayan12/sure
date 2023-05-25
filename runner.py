import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.metrics import roc_auc_score
import time

import fairness

class NeuralNetwork(nn.Module):
  def __init__(self, feature_dim, output_dim, hidden_dims_arr=[64, 32], dropout_p=0.5, no_dropout_first_layer=False):
    super().__init__()
    mod_list = []
    last_output_dim = feature_dim
    for i in range(len(hidden_dims_arr)):
      this_output_dim = hidden_dims_arr[i]
      mod_list.append(nn.Linear(last_output_dim, this_output_dim))
      mod_list.append(nn.ReLU())
      if dropout_p > 0 and (i > 0 or not no_dropout_first_layer):
        mod_list.append(nn.Dropout(dropout_p))
      last_output_dim = this_output_dim
    mod_list.append(nn.Linear(last_output_dim, output_dim))
    self.mod_list = nn.Sequential(*mod_list)
    
  def forward(self, x):
    return self.mod_list(x)

class MyDataset(Dataset):
  def __init__(self, X, X_sensitive, y):
    super().__init__()
    self.X = X
    self.X_sensitive = X_sensitive
    self.y = y

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx], self.X_sensitive[idx], idx

  def get_stats(self):
    feature_dim = self.X.shape[1]
    output_dim = len(np.unique(self.y))
    return feature_dim, output_dim

def mark_clusters(dataloader, model, device, num_pts_in_bin=30, ucb_factor=2.0):
  model.eval()
  all_best_dirs = []
  all_stress_idxs = torch.empty(0, dtype=int, device=device)
  all_okc_idxs = torch.empty(0, device=device)
  for batch, (X, y, _, idxs) in enumerate(dataloader):
    X, y = X.float().to(device), y.long().to(device)
    idxs = idxs.to(device)

    # Predict
    pred = model(X)

    # Find misclassifier points
    misc_mask = (pred.argmax(1) != y)
    if misc_mask.sum() > 0:
      ok_mask = (pred.argmax(1) == y)

      # Find clusters
      start_time_cluster = time.time()
      points_idx, best_dirs, okc_pts_idx, overall_goodness = \
          fairness.cluster_scanner(X[misc_mask], 
                                   baseline=X[ok_mask],
                                   num_pts_in_bin=num_pts_in_bin,
                                   ucb_factor=ucb_factor,
                                   device=device,
                                   )

      # Add extra loss for cluster points
      if not(len(best_dirs) == 0 or overall_goodness < -1):
        stress_ids = idxs[misc_mask][points_idx]
        okc_ids = idxs[ok_mask][okc_pts_idx]
        all_stress_idxs = torch.cat((all_stress_idxs, stress_ids))
        all_okc_idxs = torch.cat((all_okc_idxs, okc_ids))


        all_best_dirs.extend(best_dirs)

  all_stress_idxs = all_stress_idxs.cpu().numpy()
  all_okc_idxs = all_okc_idxs.cpu().numpy()
  return all_stress_idxs, all_okc_idxs

def train(dataloader, model, loss_fn, optimizer, device, wt_vec, verbose=1, label=None):
  model.train()

  wt_tot_loss = 0
  for batch, (X, y, _, idxs) in enumerate(dataloader):
    X, y, idxs = X.float().to(device), y.long().to(device), idxs.to(device)

    # Predict
    pred = model(X)

    # Standard loss
    overall_loss = loss_fn(pred, y)

    this_wt_vec = wt_vec[idxs]
    loss = (this_wt_vec * overall_loss).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wt_tot_loss += loss

  if verbose >= 1:
    print('{}wt_tot_loss={:3.3f}'.format(f'{label}: ' if label is not None else '', wt_tot_loss))

def calc_metrics(this_res_part):
  return f"macro-AUC: {roc_auc_score(this_res_part['y'], this_res_part['score'], average='macro'):2.2f}"

def test(dataloader, model, loss_fn, device, sensitive_cols, verbose):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  all_res = []
  all_sensitive_col_names, all_sensitive_col_ids = list(zip(*[x for z in list(sensitive_cols.values()) for x in z]))
  n_points = 0
  with torch.no_grad():
    for X, y, X_sensitive, _ in dataloader:
      this_res = DataFrame(X_sensitive.numpy(), columns=all_sensitive_col_names)
      this_res['y'] = y
      
      X, y = X.float().to(device), y.long().to(device)
      pred = model(X)
      this_res['loss'] = loss_fn(pred, y).cpu().numpy()
      this_res['correct'] = (pred.argmax(1) == y).type(torch.float).cpu().numpy()
      this_res['score'] = pred[:,1].type(torch.float).cpu().numpy()

      test_loss += loss_fn(pred, y).sum().item()
      correct += this_res['correct'].sum()
      n_points += len(y)
      all_res.append(this_res)

  test_loss /= n_points
  correct /= size
  all_res = pd.concat(all_res, ignore_index=True)

  if verbose > -3:
    print(f" Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:3.3f} " + calc_metrics(all_res))

  result_stats = DataFrame()
  attr_series = {}
  for s, l in sensitive_cols.items():
    for sval, _ in l:
      attr_series[sval] = s
      tmp = all_res[all_res[sval]==1]
      this_d = {'count':len(tmp),
                'accuracy':'{:2.2f}'.format(tmp['correct'].mean()),
                'macro-auc':'{:2.2f}'.format(roc_auc_score(tmp['y'], tmp['score'], average='macro')),
                'loss2':'{:3.3f}'.format(tmp['loss'].mean()),
               }
      result_stats[sval] = Series(this_d)
  attr_series = Series(attr_series)
  return result_stats, attr_series

def do_all(training_data, testing_data, sensitive_cols, num_pts_in_bin=30, cluster_loss_multiplier=1.0, min_wt_factor=1.0, batch_size=200, cluster_batch_size=5000, epochs=10, skip_epochs=3, cluster_epochs=30, seed=0, verbose=0, dropout_p=0, lr=1e-2, train_error_check_epochs=[140, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]):
  np.random.seed(seed)
  torch.manual_seed(seed)

  if cluster_batch_size < 0:
    cluster_batch_size = len(training_data) 

  shuffle=True
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
  test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=shuffle) if testing_data is not None else None
  cluster_dataloader = DataLoader(training_data, batch_size=cluster_batch_size, shuffle=shuffle)

  feature_dim, output_dim = training_data.get_stats()
  loss_fn = nn.CrossEntropyLoss(reduction='none')
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = NeuralNetwork(feature_dim=feature_dim, output_dim=output_dim, hidden_dims_arr=[64, 32], dropout_p=dropout_p).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  wt_vec = (torch.ones(len(training_data)) / len(training_data)).to(device)
  all_stress_idxs, all_okc_idxs = [], []
  all_res_train = []

  for i in range(epochs):
    if verbose >= 1:
      print(f'Iter {i}')

    if (i >= skip_epochs) and (i - skip_epochs) % cluster_epochs == 0:
      # Find stress points
      all_stress_idxs, all_okc_idxs = \
          mark_clusters(cluster_dataloader, model=model, device=device, num_pts_in_bin=num_pts_in_bin)

      # Update weights
      wt_vec2 = torch.ones(len(training_data)).to(device)
      wt_vec2[all_stress_idxs] += min_wt_factor
      wt_vec2[all_okc_idxs] += min_wt_factor
      wt_vec2 /= wt_vec2.sum()

      wt_vec = wt_vec / (1 + cluster_loss_multiplier) + (1 - 1/(1+cluster_loss_multiplier)) * wt_vec2

      # Set up the training dataloader so that enough stress+okc points are there in each mini-batch
      num_focus_pts = len(all_stress_idxs) + len(all_okc_idxs)
      effective_size = num_focus_pts
      this_batch_size = int(np.ceil(batch_size * len(training_data) / effective_size))  if num_focus_pts > 0 else batch_size
      train_dataloader = DataLoader(training_data, batch_size=this_batch_size, shuffle=shuffle)
    

    train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer,
          device=device, wt_vec=wt_vec, verbose=verbose, label=f'Iter {i}')

    if verbose >= 1 or i == epochs-1 or ((i+1) in train_error_check_epochs):
      res_train, attr_series = test(dataloader=train_dataloader, model=model, loss_fn=loss_fn, device=device, sensitive_cols=sensitive_cols, verbose=verbose)
      res_train['epoch'] = i+1
      if verbose >= 0 or (i == epochs - 1 and verbose > -2):
        print('Train error:')
        print(res_train)
      all_res_train.append(res_train)


  all_res_train = pd.concat(all_res_train, ignore_index=False)
  if test_dataloader is not None:
    res_test, attr_series = test(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device, sensitive_cols=sensitive_cols, verbose=verbose)
    res_test['epoch'] = epochs
    if verbose >= 1 or (i == epochs - 1 and verbose > -2):
      print('Test error:')
      print(res_test)

  return all_res_train, res_test, attr_series
