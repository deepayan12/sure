import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import sys
import torch
import warnings


def get_v_vc_p(bins, values, device):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    v = torch.searchsorted(bins, values)
  this_v, this_counts = torch.unique(v, return_counts=True)
  vc = torch.zeros(len(bins)+1, dtype=torch.long, device=device)
  vc[this_v] = this_counts
  p = vc / vc.sum()
  return v, vc, p

def scan_along_dir(Xv, baseline, X_so_far, points_orig_idx, this_dir, device, num_pts_in_bin=30, ucb_factor=2):
  num_bins = min(100, int(min(len(Xv), len(baseline)) / num_pts_in_bin))
  points_idx, grp_goodness, overall_goodness, extras = [], 0, 0, None

  if num_bins <= 1:
    return points_idx, grp_goodness, overall_goodness, extras

  if len(Xv) == len(baseline):
    samples = torch.cat((Xv, baseline))
  else:
    if len(Xv) < len(baseline):
      large, small = baseline, Xv
    else:
      large, small = Xv, baseline
    samples = torch.cat((small, large[torch.multinomial(torch.ones(len(large)), len(small))]))
  percentiles = torch.quantile(samples, torch.linspace(0, 1, num_bins+1).to(device))
  bins = torch.unique(percentiles[1:-1])

  num_bins = len(bins) + 1
  if num_bins < 1:
    return points_idx, grp_goodness, overall_goodness, extras

  v_baseline, vc_baseline, hatp = get_v_vc_p(bins, baseline, device)
  v, vc, p = get_v_vc_p(bins, Xv, device)
  hatp_diff = p - hatp 
  
  # Which bins are comfortably more than baseline?
  hatp_ucb = hatp_diff - ucb_factor * torch.sqrt(hatp * (1-hatp) / len(baseline) + p * (1-p) / len(Xv))
  mask = (hatp_ucb > 0)
  
  # Group the bins
  mask_shiftdown = mask.roll(1)
  mask_shiftdown[0] = False
  mask_shiftup = mask.roll(-1)
  mask_shiftup[-1] = False
  mask_up = (mask & (~mask_shiftdown))
  mask_down = (mask & (~mask_shiftup))

  hatp_diff_sum = torch.cumsum(hatp_diff * mask.type(torch.int), dim=0)
  grp_p_diff = hatp_diff_sum[mask_down] \
             - hatp_diff_sum[mask_up] \
             + hatp_diff[mask_up]
      
  if len(grp_p_diff) > 0:
    max_idx = torch.sum(mask_up[:torch.argmax(hatp_diff)+1]) - 1

    bin_idx_min, bin_idx_max = torch.where(mask_up)[0][max_idx], torch.where(mask_down)[0][max_idx]
    points_idx = torch.where((v >= bin_idx_min) & (v <= bin_idx_max))[0]
    
    grp_p_diff_max = hatp_diff[bin_idx_min:bin_idx_max+1].sum()
    grp_p_diff_tot = hatp_diff[mask].sum()
    
    grp_goodness = grp_p_diff_max
    overall_goodness = grp_p_diff_tot
    
    bin_min = bins[bin_idx_min-1] if bin_idx_min>0 else None
    bin_max = bins[bin_idx_max] if bin_idx_max<num_bins-1 else None       
    extras = {'v':v, 'vc':vc, 'hatp':hatp, 'num_bins':num_bins,
              'bin_idx_min':bin_idx_min, 'bin_idx_max':bin_idx_max,
              'bin_min':bin_min, 'bin_max':bin_max,
             }
  return points_idx, grp_goodness, overall_goodness, extras

def iterate_dirs(X, baseline, X_full, points_orig_idx, device,
                 except_dirs=np.array([]), **kwds):
  best_dir, best_points_idx, best_grp_goodness, best_overall_goodness, best_extras = None, None, None, None, None
  best_goodness = 0
  for this_dir in np.setdiff1d(np.arange(X.shape[1]), except_dirs):
    this_X_so_far = X_full[:, np.concatenate([except_dirs, [this_dir]])]
    points_idx, grp_goodness, overall_goodness, extras = scan_along_dir(
                                                  X[:,this_dir], baseline[:,this_dir],
                                                  this_X_so_far, points_orig_idx=points_orig_idx,
                                                  device=device,
                                                  this_dir=this_dir, **kwds)
    
    if grp_goodness > best_goodness and len(points_idx) < X.shape[0]:
      best_dir, best_points_idx, best_grp_goodness, best_overall_goodness, best_extras = \
              this_dir, points_idx, grp_goodness, overall_goodness, extras
      best_goodness = grp_goodness
  return best_dir, best_points_idx, best_grp_goodness, best_overall_goodness, best_extras

def cluster_scanner(X, baseline, device, **kwds):
  except_dirs = np.array([]).astype(int)
  this_X, this_baseline, points_orig_idx = X, baseline, torch.arange(X.shape[0], device=device)
  baseline_ids = torch.arange(baseline.shape[0], dtype=int, device=device)
  last_grp_goodness = 0
  last_overall_goodness = 0
  all_best_dirs = []
  while len(except_dirs) < X.shape[1]:
    best_dir, points_idx, grp_goodness, overall_goodness, extras = \
        iterate_dirs(this_X,
                     baseline[baseline_ids],
                     X_full=X,
                     points_orig_idx=points_orig_idx,
                     device=device,
                     except_dirs=except_dirs, 
                     **kwds)
    if best_dir is None or overall_goodness < last_grp_goodness:
      break
    
    # The following line works on my setup, but apparently not elsewhere.
#    to_keep_mask = (extras['bin_min'] is None or baseline[baseline_ids,best_dir]>extras['bin_min']) & \
#                   (extras['bin_max'] is None or baseline[baseline_ids,best_dir]<=extras['bin_max'])

    # Ugly workaround
    bin_min = extras['bin_min'] if (extras['bin_min'] is not None) else -1e99
    bin_max = extras['bin_max'] if (extras['bin_max'] is not None) else 1e99
    a = baseline[baseline_ids,best_dir]>bin_min
    b = baseline[baseline_ids,best_dir]<=bin_max
    to_keep_mask = a & b

    except_dirs = np.append(except_dirs, best_dir)
    this_X = X[points_idx]
    baseline_ids = baseline_ids[to_keep_mask]

    points_orig_idx = points_orig_idx[points_idx]
    last_grp_goodness = grp_goodness
    last_overall_goodness = overall_goodness
    all_best_dirs.append(best_dir)
  return points_orig_idx, all_best_dirs, baseline_ids, last_overall_goodness
