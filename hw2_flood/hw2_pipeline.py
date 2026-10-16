"""Leakage-controlled selectors for HW2. No final test/calibration labels are used."""
import hashlib
from functools import lru_cache
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SFS_CACHE={}
class FloodSelector(TransformerMixin,BaseEstimator):
    def __init__(self,method='all',k=20,selection_rows=2500,seed=42):
        self.method=method;self.k=k;self.selection_rows=selection_rows;self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X);y=np.asarray(y);p=X.shape[1]
        if self.k==p or self.method=='all':
            self.support_=np.arange(p);return self
        rng=np.random.default_rng(self.seed)
        ix=rng.choice(len(X),min(len(X),self.selection_rows),replace=False)
        A=X[ix];b=y[ix]
        if self.method=='SelectKBest':
            selected=SelectKBest(f_regression,k=self.k).fit(A,b).get_support(indices=True)
        elif self.method=='RFE':
            selected=RFE(LinearRegression(),n_features_to_select=self.k).fit(A,b).get_support(indices=True)
        elif self.method=='Lasso selection':
            reg=LassoCV(alphas=np.logspace(-6,-2,9),cv=KFold(3,shuffle=True,random_state=self.seed),max_iter=20000).fit(A,b)
            selected=np.argsort(-np.abs(reg.coef_),kind='stable')[:self.k]
        elif self.method=='Permutation':
            At,Av,bt,bv=train_test_split(A,b,test_size=.3,random_state=self.seed)
            reg=LinearRegression().fit(At,bt)
            imp=permutation_importance(reg,Av,bv,scoring='neg_mean_squared_error',n_repeats=5,random_state=self.seed,n_jobs=1)
            selected=np.argsort(-imp.importances_mean,kind='stable')[:self.k]
        elif self.method=='Forward SFS':
            key=hashlib.sha256(A.tobytes()+b.tobytes()).hexdigest()
            if key not in SFS_CACHE:
                chosen=[];remaining=list(range(p))
                inner=KFold(3,shuffle=True,random_state=self.seed)
                while len(chosen)<18:
                    scores=[]
                    for j in remaining:
                        estimator=make_pipeline(StandardScaler(),LinearRegression())
                        score=-cross_val_score(estimator,A[:,chosen+[j]],b,cv=inner,scoring='neg_mean_squared_error',error_score='raise').mean()
                        scores.append((score,j))
                    _,best=min(scores)
                    chosen.append(best);remaining.remove(best)
                SFS_CACHE[key]=chosen+remaining
            selected=SFS_CACHE[key][:self.k]
        else: raise ValueError(self.method)
        self.support_=np.sort(selected)
        return self
    def transform(self,X): return np.asarray(X)[:,self.support_]
    def get_support(self,indices=False):
        if indices:return self.support_
        mask=np.zeros(20,dtype=bool);mask[self.support_]=True;return mask
