# Feature selection schemes (CRISP-DM extension)

Schemes are predictor subsets, not 20 distinct algorithms. Exhaustive search evaluates all 31 nonempty subsets.

Business/data understanding and preparation follow analysis.py. Modeling uses LinearRegression for every subset.
Ranking uses training-only 5-fold mean MSE; smaller is better. CV scores are selection scores and may be optimistic after searching.
The holdout was already viewed in the baseline run; it is not a fresh independent confirmation. No holdout-driven tuning or reranking is performed.

Original full-model test MSE: 82010363.044301

## Top 20 by CV MSE

```text
 rank                                                                      features  k       cv_mse      cv_rmse   cv_mse_std
    1                                  R&D Spend + Administration + Marketing Spend  3 9.184690e+07  9583.678888 7.119220e+07
    2                                                   R&D Spend + Marketing Spend  2 9.375189e+07  9682.556146 7.359476e+07
    3                                                    R&D Spend + Administration  2 9.388169e+07  9689.256639 7.417348e+07
    4                 R&D Spend + Administration + Marketing Spend + State_New York  4 9.549992e+07  9772.405941 7.101828e+07
    5                  R&D Spend + Administration + Marketing Spend + State_Florida  4 9.577078e+07  9786.254853 7.345162e+07
    6                                   R&D Spend + Administration + State_New York  3 9.659780e+07  9828.418211 7.369669e+07
    7                                    R&D Spend + Administration + State_Florida  3 9.774664e+07  9886.689976 7.192808e+07
    8                                                                     R&D Spend  1 9.888057e+07  9943.871029 7.664427e+07
    9                                   R&D Spend + Marketing Spend + State_Florida  3 9.896452e+07  9948.091215 7.860134e+07
   10                                  R&D Spend + Marketing Spend + State_New York  3 9.915237e+07  9957.528464 7.484615e+07
   11 R&D Spend + Administration + Marketing Spend + State_Florida + State_New York  5 9.983623e+07  9991.808177 7.155076e+07
   12                   R&D Spend + Administration + State_Florida + State_New York  4 1.013890e+08 10069.209730 7.096794e+07
   13                  R&D Spend + Marketing Spend + State_Florida + State_New York  4 1.029817e+08 10147.991194 7.723814e+07
   14                                                     R&D Spend + State_Florida  2 1.036813e+08 10182.401969 7.542653e+07
   15                                                    R&D Spend + State_New York  2 1.040456e+08 10200.276255 7.605793e+07
   16                                    R&D Spend + State_Florida + State_New York  3 1.091681e+08 10448.354181 7.367095e+07
   17                                              Administration + Marketing Spend  2 7.077342e+08 26603.274364 5.146143e+08
   18                                                               Marketing Spend  1 7.597475e+08 27563.516731 5.456090e+08
   19                              Administration + Marketing Spend + State_Florida  3 8.028222e+08 28334.117800 6.257767e+08
   20                             Administration + Marketing Spend + State_New York  3 8.124019e+08 28502.664174 5.945037e+08
```

## Top 5 holdout evaluation

```text
 rank                                                      features  k       cv_mse     cv_rmse   cv_mse_std     test_mse   test_rmse    test_mae  test_r2
    1                  R&D Spend + Administration + Marketing Spend  3 9.184690e+07 9583.678888 7.119220e+07 8.092632e+07 8995.905803 6979.152252 0.900065
    2                                   R&D Spend + Marketing Spend  2 9.375189e+07 9682.556146 7.359476e+07 6.734383e+07 8206.328813 6469.181721 0.916838
    3                                    R&D Spend + Administration  2 9.388169e+07 9689.256639 7.417348e+07 8.376413e+07 9152.274782 7181.327132 0.896561
    4 R&D Spend + Administration + Marketing Spend + State_New York  4 9.549992e+07 9772.405941 7.101828e+07 8.127225e+07 9015.112542 6974.873066 0.899638
    5  R&D Spend + Administration + Marketing Spend + State_Florida  4 9.577078e+07 9786.254853 7.345162e+07 8.201037e+07 9055.957752 6961.486846 0.898727
```

Deployment: saved evaluation models use 40 training rows; deployment models are refit on all 50 rows and have no fresh test score.
