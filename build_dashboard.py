"""Build an offline interactive dashboard from fitted sklearn models."""
from pathlib import Path
import ast, json
import numpy as np
import pandas as pd
from scipy.stats import t, f
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from feature_selection import make_model, NUM, NAMES

def build(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    df=pd.read_csv(root/'data/50_Startups.csv')
    Xtr,Xte,ytr,yte=train_test_split(df[NUM+['State']],df.Profit,test_size=.2,random_state=42)
    ranking=pd.read_csv(root/'outputs/all_31_schemes.csv')
    models=[]
    for row in ranking.itertuples():
        ix=list(ast.literal_eval(row.indices))
        model=make_model(ix).fit(Xtr,ytr)
        reg=model.named_steps['regression']
        A=model[:-1].transform(Xtr)
        design=np.column_stack([np.ones(len(A)),A])
        resid=ytr.to_numpy()-model.predict(Xtr)
        dof=len(A)-np.linalg.matrix_rank(design)
        cov=np.linalg.pinv(design.T@design)*(resid@resid/dof)
        se=np.sqrt(np.maximum(np.diag(cov),0))
        beta=np.r_[reg.intercept_,reg.coef_]
        ts=np.divide(beta,se,out=np.zeros_like(beta),where=se>0)
        pv=2*t.sf(np.abs(ts),dof)
        crit=t.ppf(.975,dof)
        vifs=[]
        from sklearn.linear_model import LinearRegression
        for j in range(len(ix)):
            if len(ix)==1: vifs.append(1.)
            else:
                r=LinearRegression().fit(np.delete(A,j,axis=1),A[:,j]).score(np.delete(A,j,axis=1),A[:,j])
                vifs.append(float(1/max(1-r,1e-12)))
        pred=model.predict(Xte)
        mse=mean_squared_error(yte,pred)
        r2=r2_score(yte,pred)
        rtrain=model.score(Xtr,ytr)
        fs=(rtrain/len(ix))/((1-rtrain)/dof)
        rows=[{'id':int(idx), 'actual':float(a),'predicted':float(p),'residual':float(a-p), 'pct':float(abs(a-p)/abs(a)*100)} for idx,a,p in zip(yte.index,yte,pred)]
        rows.sort(key=lambda r:abs(r['residual']),reverse=True)
        models.append({'rank':int(row.rank),'indices':ix,'features':[NAMES[i] for i in ix], 'k':len(ix),
          'cv_mse':float(row.cv_mse),'mse':float(mse),'rmse':float(np.sqrt(mse)),
          'mae':float(mean_absolute_error(yte,pred)),'r2':float(r2),
          'adjusted_r2':float(1-(1-r2)*(len(yte)-1)/(len(yte)-len(ix)-1)),
          'f':float(fs),'f_p':float(f.sf(fs,len(ix),dof)),
          'beta':beta.tolist(),'se':se.tolist(),'p':pv.tolist(),'t':ts.tolist(),
          'ci':[[float(b-crit*s),float(b+crit*s)] for b,s in zip(beta,se)], 'vif':[None]+vifs,'rows':rows})
    # Five selection algorithms operate only on training data.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.feature_selection import RFE, SequentialFeatureSelector
    raw=make_model(range(5)).named_steps['prepare'].fit_transform(Xtr)
    scaled=StandardScaler().fit_transform(raw)
    from sklearn.model_selection import KFold
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    lasso=LassoCV(cv=cv,max_iter=100000).fit(scaled,ytr)
    elastic=ElasticNetCV(l1_ratio=[.1,.5,.9,1.],cv=cv,max_iter=100000).fit(scaled,ytr)
    algorithm_indices={name:[] for name in ['Lasso','Forward SFS','RFE','ElasticNet','Exhaustive Search']}
    for k in range(1,6):
        for name,weights in [('Lasso',lasso.coef_),('ElasticNet',elastic.coef_)]:
            algorithm_indices[name].append(sorted(np.argsort(-np.abs(weights),kind='stable')[:k].tolist()))
        algorithm_indices['RFE'].append(sorted(np.flatnonzero(RFE(LinearRegression(),n_features_to_select=k).fit(scaled,ytr).support_).tolist()))
        if k==5: forward=list(range(5))
        else:
            estimator=Pipeline([('scale',StandardScaler()),('regression',LinearRegression())])
            selector=SequentialFeatureSelector(estimator,n_features_to_select=k,direction='forward',scoring='neg_mean_squared_error',cv=cv)
            forward=sorted(np.flatnonzero(selector.fit(raw,ytr).get_support()).tolist())
        algorithm_indices['Forward SFS'].append(forward)
        algorithm_indices['Exhaustive Search'].append(next(m['indices'] for m in models if m['k']==k))
    algorithms={name:[next(i for i,m in enumerate(models) if m['indices']==ix) for ix in subsets] for name,subsets in algorithm_indices.items()}
    payload={'algorithms':algorithms,'models':models,'names':NAMES,'ranges':{col:[float(df[col].min()),float(df[col].max())] for col in NUM}}
    html=(root/'dashboard_template.html').read_text(encoding='utf-8').replace('__MODEL_DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False))
    (root/'dashboard.html').write_text(html,encoding='utf-8')
    (root/'outputs/dashboard_models.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print('Dashboard created:',root/'dashboard.html')
    return payload
if __name__=='__main__': build()



