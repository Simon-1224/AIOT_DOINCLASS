# -*- coding: utf-8 -*-
"""5115056023 HW2: CRISP-DM + linear regression + feature selection.
Run: python 5115056023_hw2.py
Evidence integration/report: python build_report.py ; python package_submission.py
"""
from pathlib import Path
import json, hashlib, platform, time
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from hw2_pipeline import FloodSelector

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SEED=42
LABELS=['季風強度','地形排水','河川管理','森林砍伐','都市化','氣候變遷','水壩品質','淤積','農業作法','侵占河道','災害準備不足','排水系統','沿海脆弱度','山崩','集水區','基礎設施劣化','人口分數','濕地流失','規劃不足','政治因素']
METHODS=['SelectKBest','RFE','Forward SFS','Lasso selection','Permutation']

def savejson(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def metrics(y,p,k):
    r=float(r2_score(y,p));n=len(y)
    return {'MAE':float(mean_absolute_error(y,p)),'MSE':float(mean_squared_error(y,p)),
      'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':r,
      'Adjusted_R2':float(1-(1-r)*(n-1)/(n-k-1)) if k<n-1 else None,
      'outside_0_1':int(((p<0)|(p>1)).sum())}
def pipeline(method,k,reg=None):
    return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),
      ('select',FloodSelector(method,k)),('regression',reg if reg is not None else LinearRegression())])
def figure(name):plt.tight_layout();plt.savefig(FIG/name,dpi=170,bbox_inches='tight');plt.close()

def main():
    start=time.monotonic()
    # 1 Business Understanding / 2 Data Understanding
    source=json.loads((ROOT/'data/source.json').read_text(encoding='utf-8'))
    raw=pd.read_csv(ROOT/'data/flood.csv')
    names=[c for c in raw if c!='FloodProbability']
    assert len(names)==20 and raw.shape==(50000,21)
    actual_hash=hashlib.sha256((ROOT/'data/flood.csv').read_bytes()).hexdigest()
    assert actual_hash==source['sha256'],'Dataset changed: update documented source before running'
    assert raw.FloodProbability.notna().all() and raw.FloodProbability.between(0,1).all()
    quality={'rows':len(raw),'features':20,'columns':21,'missing':raw.isna().sum().to_dict(),
      'duplicate_rows':int(raw.duplicated().sum()),'target_range':[float(raw.FloodProbability.min()),float(raw.FloodProbability.max())],
      'dtypes':{c:str(t) for c,t in raw.dtypes.items()}}
    if quality['duplicate_rows']:raise ValueError('Handle duplicates before splitting')
    # 3 Preparation: test 20%, calibration 20%, development 60%.
    # No EDA involving target, tuning or feature selection uses calibration/test.
    devcal,test=train_test_split(raw.index.to_numpy(),test_size=.2,random_state=SEED)
    dev,cal=train_test_split(devcal,test_size=.25,random_state=SEED)
    X=raw[names];y=raw.FloodProbability
    Xd,yd=X.loc[dev],y.loc[dev];Xc,yc=X.loc[cal],y.loc[cal];Xt,yt=X.loc[test],y.loc[test]
    assert not(set(dev)&set(cal) or set(dev)&set(test) or set(cal)&set(test))
    savejson(OUT/'split_indices.json',{'development':dev.tolist(),'calibration':cal.tolist(),'test':test.tolist()})
    savejson(OUT/'data_quality.json',quality)
    Xd.join(yd).describe().to_csv(OUT/'development_description.csv')
    pd.DataFrame({'feature':names,'zh':LABELS,'dtype':[str(raw[c].dtype) for c in names],
       'dev_min':Xd.min().values,'dev_max':Xd.max().values,'dev_mean':Xd.mean().values}).to_csv(OUT/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    corr=Xd.join(yd).corr();corr.to_csv(OUT/'development_correlation.csv')
    plt.figure(figsize=(11,9));plt.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');plt.colorbar(label='Pearson r')
    plt.xticks(range(21),corr.columns,rotation=90,fontsize=7);plt.yticks(range(21),corr.columns,fontsize=7);plt.title('Development data only: feature correlations');figure('01_correlation.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));axes[0].hist(yd,bins=40,color='#167f8d');axes[0].set(title='Development target distribution',xlabel='FloodProbability',ylabel='Count')
    targetcorr=corr.FloodProbability.drop('FloodProbability').sort_values();axes[1].barh(targetcorr.index,targetcorr,color='#167f8d');axes[1].tick_params(axis='y',labelsize=6);axes[1].set(title='Development feature-target correlations',xlabel='Pearson r');figure('02_eda.png')
    # 4 Modeling: selection is refitted INSIDE each outer CV training fold.
    cv=KFold(5,shuffle=True,random_state=SEED)
    specs={}
    specs['Mean baseline']={'method':'all','k':20,'reg':DummyRegressor()}
    specs['OLS full']={'method':'all','k':20,'reg':LinearRegression()}
    for alpha in [.01,.1,1.,10.,100.,1000.]:specs[f'Ridge alpha={alpha:g}']={'method':'all','k':20,'reg':Ridge(alpha=alpha)}
    for alpha in [1e-6,1e-5,1e-4,1e-3,.01]:specs[f'Lasso alpha={alpha:g}']={'method':'all','k':20,'reg':Lasso(alpha=alpha,max_iter=20000)}
    for method in METHODS:
        for k in [14,16,18,20]:specs[f'{method} k={k}']={'method':method,'k':k,'reg':LinearRegression()}
    # Identical folds, same target and same resources; not part of required linear-model ranking.
    specs['Histogram gradient boosting (comparison)']={'method':'all','k':20,'reg':HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,random_state=SEED)}
    scores=[];folds=[];fit_models={}
    for name,spec in specs.items():
        model=pipeline(spec['method'],spec['k'],clone(spec['reg']))
        result=cross_validate(model,Xd,yd,cv=cv,scoring={'mse':'neg_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'},error_score='raise')
        row={'scheme':name,'method':spec['method'],'k':spec['k'],'CV_MSE':float(-result['test_mse'].mean()),
          'CV_MSE_SD':float(result['test_mse'].std(ddof=1)),'CV_RMSE':float(np.sqrt(-result['test_mse'].mean())),
          'CV_MAE':float(-result['test_mae'].mean()),'CV_R2':float(result['test_r2'].mean())}
        scores.append(row)
        for fold in range(5):folds.append({'scheme':name,'fold':fold+1,'MSE':float(-result['test_mse'][fold]),'MAE':float(-result['test_mae'][fold]),'R2':float(result['test_r2'][fold])})
        print(f'{name}: CV MSE={row["CV_MSE"]:.9f}',flush=True)
    rank=pd.DataFrame(scores).sort_values(['CV_MSE','k','scheme']).reset_index(drop=True)
    rank.insert(0,'rank',np.arange(1,len(rank)+1))
    rank.to_csv(OUT/'cv_ranking.csv',index=False);pd.DataFrame(folds).to_csv(OUT/'cv_folds.csv',index=False)
    eligible=rank[~rank.scheme.isin(['Mean baseline','Histogram gradient boosting (comparison)'])]
    winner=str(eligible.iloc[0].scheme)
    if float(rank.set_index('scheme').loc['OLS full','CV_MSE']) == float(eligible.iloc[0].CV_MSE): winner='OLS full'
    savejson(OUT/'selection_decision.json',{'winner':winner,'criterion':'Smallest development-only 5-fold mean MSE among linear schemes','test_used_for_selection':False,'calibration_used_for_selection':False})
    print('Locked selected linear model:',winner,flush=True)
    # 5 Evaluation: evaluate once after freezing the ranking. Never rerank using test scores.
    records=[];selected_features={};web_models=[]
    for name,spec in specs.items():
        model=pipeline(spec['method'],spec['k'],clone(spec['reg'])).fit(Xd,yd)
        fit_models[name]=model
        pred=model.predict(Xt)
        k=0 if name=='Mean baseline' else len(model.named_steps['select'].support_)
        m=metrics(yt,pred,k)
        # Adjusted R2 uses predictor count; for penalized/nonlinear models it is heuristic, not effective df.
        records.append({'scheme':name,'k':k,**m})
        support=model.named_steps['select'].support_
        selected_features[name]=[names[i] for i in support]
        reg=model.named_steps['regression']
        if hasattr(reg,'coef_'):
            scale=model.named_steps['scale'];coeff=np.zeros(20);coeff[support]=np.asarray(reg.coef_)/scale.scale_[support]
            intercept=float(reg.intercept_-np.sum(np.asarray(reg.coef_)*scale.mean_[support]/scale.scale_[support]))
            web_models.append({'scheme':name,'features':selected_features[name],'k':k,'metrics':m,'coefficients':coeff.tolist(),'intercept':intercept})
    test_table=pd.DataFrame(records).merge(rank[['scheme','rank','CV_MSE','CV_RMSE']],on='scheme').sort_values('rank')
    test_table.to_csv(OUT/'test_metrics.csv',index=False)
    savejson(OUT/'selected_features.json',selected_features)
    chosen=fit_models[winner];prediction=chosen.predict(Xt)
    # Split conformal: fit on development only; calibrate absolute residuals on independent calibration.
    cal_res=np.abs(yc.to_numpy()-chosen.predict(Xc));ncal=len(cal_res)
    quantile_rank=int(np.ceil((ncal+1)*.95));q=float(np.sort(cal_res)[quantile_rank-1])
    lower=prediction-q;upper=prediction+q
    interval={'method':'split conformal absolute residual','nominal_coverage':.95,'calibration_rows':ncal,
       'quantile_order':quantile_rank,'radius':q,'test_coverage':float(np.mean((yt>=lower)&(yt<=upper))),
       'mean_width':float(np.mean(upper-lower)),'clipped':False,
       'clipped_metrics':metrics(yt,np.clip(prediction,0,1),len(selected_features[winner])),
       'assumptions':'Exchangeability of calibration and future observations; marginal, not conditional coverage; no geographical or temporal validation available.'}
    savejson(OUT/'prediction_interval.json',interval)
    # Verify a potential generated-target relationship using development data only.
    sum_model=LinearRegression().fit(Xd.sum(axis=1).to_numpy().reshape(-1,1),yd)
    relation={'scope':'development data only','sum_slope':float(sum_model.coef_[0]),'intercept':float(sum_model.intercept_),
      'max_abs_error':float(np.max(np.abs(yd-sum_model.predict(Xd.sum(axis=1).to_numpy().reshape(-1,1))))),
      'interpretation':'Target is an effectively deterministic linear sum in this downloaded version. This does not validate real-world flood prediction.'}
    savejson(OUT/'target_relationship.json',relation)
    # Prespecified reduced-model illustration; not a new winner selected from holdout scores.
    reduced=fit_models['SelectKBest k=14']
    rp=reduced.predict(Xt);rc=np.abs(yc.to_numpy()-reduced.predict(Xc))
    rq=float(np.sort(rc)[quantile_rank-1]);rl=rp-rq;ru=rp+rq
    auxiliary={'scheme':'SelectKBest k=14','radius':rq,'test_coverage':float(np.mean((yt>=rl)&(yt<=ru))),
      'mean_width':2*rq,'role':'illustration of information loss and a nondegenerate interval; not selected winner'}
    savejson(OUT/'reduced_prediction_interval.json',auxiliary)
    rshow=np.argsort(rp)[np.linspace(0,len(rp)-1,120,dtype=int)]
    plt.figure(figsize=(11,4));plt.fill_between(np.arange(120),rl[rshow],ru[rshow],alpha=.22,color='#167f8d',label='95% split-conformal prediction interval');plt.plot(rp[rshow],label='Prediction',color='#167f8d');plt.scatter(np.arange(120),yt.to_numpy()[rshow],s=12,color='#e28c42',label='Observed');plt.title(f'Reduced 14-feature illustration: test coverage {auxiliary["test_coverage"]:.2%}, width {2*rq:.5f}');plt.xlabel('120 test samples, sorted by prediction (not time)');plt.ylabel('FloodProbability');plt.legend(fontsize=8);figure('09_reduced_prediction_intervals.png')
    # Exploratory post-selection diagnostics for a nondegenerate reduced OLS.
    rz=reduced[:-1].transform(Xd);rd=np.column_stack([np.ones(len(rz)),rz]);rf=reduced.predict(Xd);rr=yd.to_numpy()-rf
    rex=np.column_stack([rd,rf**2,rf**3]);rb=np.linalg.lstsq(rex,yd,rcond=None)[0]
    rsse=float(rr@rr);rss2=float(np.sum((yd-rex@rb)**2));rdf=len(yd)-np.linalg.matrix_rank(rex)
    rreset=max(0.,((rsse-rss2)/2)/(rss2/rdf));rbp=float(len(rz)*LinearRegression().fit(rz,rr**2).score(rz,rr**2))
    rjb=stats.jarque_bera(rr);rord=rr[np.argsort(dev)]
    reduced_diagnostics={'model':'SelectKBest k=14','scope':'Development-only, post-selection exploratory diagnostics; p-values do not account for feature selection.',
       'RESET_F':float(rreset),'RESET_p':float(stats.f.sf(rreset,2,rdf)),
       'BP_LM':rbp,'BP_p':float(stats.chi2.sf(rbp,14)),
       'JB':float(rjb.statistic),'JB_p':float(rjb.pvalue),
       'DW_row_order_descriptive_only':float(np.sum(np.diff(rord)**2)/np.sum(rord**2)),
       'independence':'No valid temporal/spatial ordering: independence cannot be established.'}
    savejson(OUT/'reduced_diagnostics.json',reduced_diagnostics)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));axes[0].scatter(rf,rr,s=3,alpha=.1);axes[0].axhline(0,color='black',ls='--');axes[0].set(xlabel='Fitted',ylabel='Residual',title='Reduced OLS development residuals');axes[1].hist(rr,bins=45,color='#167f8d');axes[1].set(title='Reduced OLS residual distribution',xlabel='Residual');stats.probplot(rr,dist='norm',plot=axes[2]);axes[2].set_title('Reduced OLS Q-Q plot');figure('10_reduced_assumptions.png')
    predictions=pd.DataFrame({'row_id':test,'actual':yt.to_numpy(),'predicted':prediction,'lower_95':lower,'upper_95':upper,'residual':yt.to_numpy()-prediction})
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    joblib.dump(chosen,OUT/'selected_evaluation_model.joblib')
    joblib.dump({'model':chosen,'radius':q,'features':names,'schema':'development model + calibration radius'},OUT/'prediction_bundle.joblib')
    # Separate OLS full-model diagnostics on development data; not post-selection inference.
    ols=fit_models['OLS full'];Z=ols[:-1].transform(Xd);D=np.column_stack([np.ones(len(Z)),Z])
    residual=yd.to_numpy()-ols.predict(Xd);fitted=ols.predict(Xd)
    r2base=r2_score(yd,fitted)
    extra=np.column_stack([D,fitted**2,fitted**3])
    b=np.linalg.lstsq(extra,yd,rcond=None)[0];sseextra=np.sum((yd-extra@b)**2);sse=np.sum(residual**2)
    reset=max(0.,((sse-sseextra)/2)/(sseextra/(len(yd)-np.linalg.matrix_rank(extra))))
    bpreg=LinearRegression().fit(Z,residual**2);bp=float(len(Z)*bpreg.score(Z,residual**2))
    jb=stats.jarque_bera(residual)
    order=np.argsort(dev);ordered=residual[order];dw=float(np.sum(np.diff(ordered)**2)/np.sum(ordered**2))
    vif=[]
    for j in range(20):
        r=LinearRegression().fit(np.delete(Z,j,axis=1),Z[:,j]).score(np.delete(Z,j,axis=1),Z[:,j])
        vif.append(float(1/(1-r)))
    pd.DataFrame({'feature':names,'VIF':vif}).to_csv(OUT/'vif.csv',index=False)
    diagnostics={'scope':'OLS full model, development data only. No inferential claim for selected/penalized models.',
      'linearity_RESET_F':float(reset),'linearity_RESET_p':float(stats.f.sf(reset,2,len(yd)-np.linalg.matrix_rank(extra))),
      'heteroskedasticity_BP_LM':bp,'heteroskedasticity_BP_p':float(stats.chi2.sf(bp,20)),
      'normality_JB':float(jb.statistic),'normality_JB_p':float(jb.pvalue),
      'durbin_watson_original_row_order':dw,'independence_limit':'Row order is not temporal/spatial. DW is descriptive only; independence cannot be established.',
      'max_VIF':max(vif),'full_design_rank':int(np.linalg.matrix_rank(D)),'condition_number_standardized':float(np.linalg.cond(D)),
      'interpretation':'Diagnostics can reveal departures but do not prove assumptions. Large sample p values can flag small effects.'}
    diagnostics['residual_sd']=float(np.std(residual))
    diagnostics['numerical_zero_residual']=bool(np.max(np.abs(residual))<1e-12)
    if diagnostics['numerical_zero_residual']:
        for key in ['linearity_RESET_F','linearity_RESET_p','heteroskedasticity_BP_LM','heteroskedasticity_BP_p','normality_JB','normality_JB_p','durbin_watson_original_row_order']: diagnostics[key]=None
        diagnostics['interpretation']='Residuals are at floating-point precision. RESET/BP/JB/DW inference is not meaningful and is intentionally suppressed. VIF and rank remain descriptive. Independence cannot be established.'
    savejson(OUT/'diagnostics.json',diagnostics)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(yt,prediction,s=6,alpha=.2,color='#167f8d');axes[0].plot([.25,.75],[.25,.75],'k--');axes[0].set(xlabel='Actual FloodProbability',ylabel='Predicted',title='Final holdout: actual vs predicted')
    axes[1].scatter(prediction,yt-prediction,s=6,alpha=.2,color='#167f8d');axes[1].axhline(0,color='black',ls='--');axes[1].set(xlabel='Predicted',ylabel='Actual - predicted',title='Final holdout residuals');figure('03_test_predictions.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5));axes[0].hist(residual,bins=45,color='#167f8d');axes[0].set(title='Development OLS residuals',xlabel='Residual');stats.probplot(residual,dist='norm',plot=axes[1]);axes[1].set_title('OLS residual Q-Q plot');axes[2].scatter(fitted,residual,s=3,alpha=.1);axes[2].axhline(0,color='black',ls='--');axes[2].set(xlabel='OLS fitted',ylabel='Residual',title='OLS development residual pattern');figure('04_assumptions.png')
    # The interval plot is ordered by prediction solely for readability, not a time-series claim.
    show=np.argsort(prediction)[np.linspace(0,len(prediction)-1,120,dtype=int)];px=np.arange(len(show))
    plt.figure(figsize=(11,4));plt.fill_between(px,lower[show],upper[show],alpha=.22,color='#167f8d',label='95% split-conformal prediction interval');plt.plot(px,prediction[show],color='#167f8d',lw=1.6,label='Prediction');plt.scatter(px,yt.to_numpy()[show],s=12,color='#e28c42',label='Observed');plt.xlabel('120 holdout samples, sorted by prediction (not time)');plt.ylabel('FloodProbability');plt.title(f'Coverage on all {len(test):,} test rows: {interval["test_coverage"]:.2%}; mean width {interval["mean_width"]:.5f}');plt.legend(fontsize=8);figure('05_prediction_intervals.png')
    plt.figure(figsize=(9,4.5))
    for method in METHODS:
        part=rank[rank.method==method].sort_values('k');plt.plot(part.k,part.CV_RMSE,'o-',label=method)
    plt.xticks([14,16,18,20]);plt.xlabel('Selected feature count');plt.ylabel('Development 5-fold CV RMSE');plt.title('Selection refitted inside every CV training fold');plt.legend(fontsize=8);figure('06_feature_selection.png')
    keep=['Mean baseline','OLS full']+[rank[rank.scheme.str.startswith(p)].iloc[0].scheme for p in ['Ridge alpha=','Lasso alpha=']]+['Histogram gradient boosting (comparison)']
    tab=test_table.set_index('scheme').loc[keep]
    fig,axes=plt.subplots(1,2,figsize=(11,4));short=['Mean','OLS','Ridge','Lasso','HistGB'];axes[0].bar(short,tab.CV_RMSE,color='#167f8d');axes[0].set(title='Development CV RMSE',ylabel='RMSE');axes[1].bar(short,tab.RMSE,color='#e28c42');axes[1].set(title='Holdout RMSE (no reranking)',ylabel='RMSE');figure('07_model_comparison.png')
    plt.figure(figsize=(10,4));plt.barh(names,vif,color='#167f8d');plt.xlabel('VIF');plt.title('Full OLS: development feature multicollinearity');plt.yticks(fontsize=7);figure('08_vif.png')
    for m in web_models:
        m.update({'CV_MSE':float(rank.set_index('scheme').loc[m['scheme'],'CV_MSE'])})
    payload={'student_id':'5115056023','source':source,'quality':quality,'features':names,'labels':LABELS,
      'development_rows':len(dev),'calibration_rows':len(cal),'test_rows':len(test),'winner':winner,'interval':interval,
      'ranges':[[float(Xd[c].min()),float(Xd[c].max())] for c in names],'defaults':[float(v) for v in Xd.median()],
      'models':web_models,'target_relationship':relation,'reduced_interval':auxiliary,'diagnostics':diagnostics,'test_points':predictions.iloc[show].to_dict(orient='records')}
    savejson(OUT/'dashboard_data.json',payload)
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__}
    manifest={'student_id':'5115056023','date':'2026-10-16','seed':SEED,'source':source,'split':{'development':len(dev),'calibration':len(cal),'test':len(test)},
      'selection_rows_cap_per_fold':2500,'selection_sampling':'deterministic random subset of each CV training fold only; final regressors use full fold',
      'candidate_schemes':len(specs),'winner':winner,'winner_test':test_table.set_index('scheme').loc[winner].to_dict(),
      'interval':interval,'diagnostics':diagnostics,'versions':versions,'runtime_seconds':time.monotonic()-start,
      'completion':'Model analysis complete; NotebookLM and original conversation PDFs must be supplied separately.'}
    savejson(OUT/'run_manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=True,indent=2),flush=True)
    # Build UI from exported data, with no Python execution required on Pages.
    template=ROOT/'web/template.html'
    if template.exists():(ROOT/'web/index.html').write_text(template.read_text(encoding='utf-8').replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False)),encoding='utf-8')
    import shutil
    if template.exists(): shutil.copytree(FIG,ROOT/'web/figures',dirs_exist_ok=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()


