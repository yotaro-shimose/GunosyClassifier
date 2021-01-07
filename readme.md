
# 精度評価
２つの手法についてAccuracy, Confusion Matrixによる評価を行った

## BertFeatureClassifier

Command: python manage.py evaluate_bfc

accuracy: 0.8805

confusion_matrix: 

```math
[[328   7   3   3   1   2   4   2]
 [  5 425   0   2   0   1   0   0]
 [  8   0  97   0   0   7   1   2]
 [  6  16   0  67   8   6  14   0]
 [ 12   4   0  10  43   0   5   0]
 [  8   1   0   2   1 126   3   4]
 [ 11   0   4   2   1   2 161   2]
 [ 10   1   0   0   0   4   4 146]]
 ```