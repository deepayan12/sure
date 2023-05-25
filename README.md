# Robust, Explainable, and Fair Classification without Sensitive Attributes (SURE)
### Train a classifier to be accurate on all subgroups within a dataset

Sample run:
<pre>
/usr/bin/python3 -u test.py --dataset example --dropout_p 0.5 --epochs 2000 --num_procs 5 --num_seeds 5

==== Utility = worst-accuracy among all groups based on sim-subgroups (baseline, subgroup at 0 degrees, subgroup at 90 degrees)
Train/Test  Test Train
epoch       2000  50    100   150   200   300   800   1300  1800  2000
Method      SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE
seed
0           0.57  0.45  0.49  0.57  0.53  0.51  0.52  0.57  0.56  0.58
1           0.57  0.49  0.51  0.53  0.49  0.49  0.49  0.50  0.49  0.59
2           0.75  0.47  0.47  0.55  0.51  0.59  0.70  0.74  0.74  0.75
3           0.74  0.49  0.46  0.59  0.49  0.57  0.59  0.69  0.73  0.73
4           0.50  0.45  0.45  0.49  0.52  0.45  0.55  0.56  0.56  0.56
Train/Test  Test Train
epoch       2000  50    100   150   200   300   800   1300  1800  2000
Method      SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE  SURE
count       5.00  5.00  5.00  5.00  5.00  5.00  5.00  5.00  5.00  5.00
mean        0.63  0.47  0.48  0.55  0.51  0.52  0.57  0.61  0.62  0.64
std         0.11  0.02  0.02  0.04  0.02  0.06  0.08  0.10  0.11  0.09
min         0.50  0.45  0.45  0.49  0.49  0.45  0.49  0.50  0.49  0.56
25%         0.57  0.45  0.46  0.53  0.49  0.49  0.52  0.56  0.56  0.58
50%         0.57  0.47  0.47  0.55  0.51  0.51  0.55  0.57  0.56  0.59
75%         0.74  0.49  0.49  0.57  0.52  0.57  0.59  0.69  0.73  0.73
max         0.75  0.49  0.51  0.59  0.53  0.59  0.70  0.74  0.74  0.75
</pre>

The file `example.npz` contains 7 pieces of information:
* sensitive_cols, 
* X_train and X_test, 
* X_train_sensitive and X_test_sensitive, and
* y_train and y_test

Description:
* X_train is a matrix where each row is the feature vector of one data point.
* y_train is a 1D array of 0/1
* X_train_sensitive is a matrix where the rows are data points, the columns are the various groups/subgroups, and each cell is 1 if the datapoint belongs to the given subgroup and 0 otherwise.
* sensitive_cols is a dictionary that describes the subgroups. For example:
  * `sensitive_cols={'gender':[('Male',0), ('Female',1), ('Other', 2)]}` has three groups for gender named Male/Female/Other. So X_train_sensitive has 3 columns.
  * `sensitive_cols={'gender':[('Male',0), ('Female',1), ('Other', 2)], 'race':[('Caucasian', 3), ('African-American', 4), ('Other',5)]}` has three groups for gender and three for race. So X_train_sensitive has 6 columns.

If you use this code, please cite the following paper
<pre>
SURE: Robust, Explainable, and Fair Classification without Sensitive Attributes,
by D. Chakrabarti,
in KDD 2023.
</pre>
