#!/usr/bin/env python
#######################################################################################
###### Copy right St. Jude Children's Research Hospital 2020 ##########################
###### Author: Ramzi Alsallaq (ramzi.alsallaq@stjude.org)    ##########################
#######################################################################################
import sys, os, glob
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles
from statsmodels.duration.hazard_regression import PHReg
import statsmodels.api as sm
import pylab as pl
import seaborn as sbs
import _pickle as pickle
sbs.set(context='talk', style='darkgrid', palette='deep', font='sans-serif')
from sklearn.metrics import auc
from scipy.interpolate import interp1d
import pymc3 as pm3
import theano
log_filename = 'csp_run.log'
logging.basicConfig(filename=log_filename,filemode='w',level=logging.DEBUG,format='[%(asctime)s-%(levelname)s] %(message)s',datefmt='%d-%b-%y %H:%M:%S')
logging.info('The output will be logged into file {}'.format(log_filename))
logger = logging.getLogger(__name__)

def cleanData(dataF, minimumNumOfPoints, lowestDetectableCFU=None):
    logger.info('filter away those with missing or zero initial inoculum')
    numberBefore = dataF.shape[0]
    initialInnocCol_ind = dataF.columns.isin([0,'0'])
    missingB0_ind = dataF.loc[:,initialInnocCol_ind].isnull()
    logger.info("a total of {} is missing initial inoculum and will be dropped".format(missingB0_ind.sum()))
    zeroB0_ind = dataF.loc[:,initialInnocCol_ind]==0.0
    logger.info("a total of {} has zero initial inoculum and will be dropped".format(zeroB0_ind.sum()))
    todrop=(missingB0_ind|zeroB0_ind)
    dataF = dataF.loc[~todrop.values.flatten(),:]
    logger.info("a total of {} is dropped, a total of {} is kept".format(numberBefore-dataF.shape[0],dataF.shape[0]))
    logger.info("filter away those with <= {} points".format(minimumNumOfPoints))
    indKeep = (~dataF.isnull()).sum(axis=1) > minimumNumOfPoints
    dataF = dataF.loc[indKeep,:]
    logger.info("A total of {} is dropped, a total of {} is kept".format(dataF.shape[0]-indKeep.sum(),dataF.shape[0]))
    if lowestDetectableCFU is not None:
        logger.info("replace values less than the minimum detectable density by the minimum detectable density {}".format(lowestDetectableCFU))
        dataF[dataF<lowestDetectableCFU]=lowestDetectableCFU

    return dataF

def addAUCs(inDF, concs, uselog10=True):
    df=inDF.loc[:,concs]
    xx=df.columns.astype(float)
    for i, row in df.iterrows():
        yy=row.astype(float)
        ind_ = np.where(~yy.isnull())[0]
        if uselog10: #wotton auc under log10 counts
            log10values=np.where(yy[ind_]>0,np.log10(yy[ind_]), 0)
            ff = interp1d(xx[ind_],log10values)
            inDF.loc[i,'aucFullLog10']=auc(xx[ind_], ff(xx[ind_]))
        else:
            ff=interp1d(xx[ind_],yy[ind_])
            inDF.loc[i,'aucFull']=auc(xx[ind_], ff(xx[ind_]))
    return inDF


def refStrainProfiling(refPAPs, regselect, maxConc):
    def glm_linear(x_, y_):
        """ Regressing AUC under counts on B0 in the log space"""
        x=np.array(x_).copy()
        y=np.array(y_).copy()
        logger.info("x_values = {}".format(x))
        logger.info("y_values = {}".format(y))
        #using probabilistic programing
        with pm3.Model() as model1:
            #define priors
            sigma = pm3.HalfCauchy('sigma', beta=1e7, testval=1.e7)
            intercept = pm3.Normal('intercept',0, tau=1./1e8**2, shape=x.shape[0])
            cumSurv = pm3.Normal('cumSurv',0,tau=1/1e5**2)
            liklelihood = pm3.Normal('y',mu=intercept + cumSurv*x, tau=1./sigma**2, observed=y)
    
            #inference
            logger.info("Sampling for inference")
            trace  = pm3.sample(1000, tune=8000, cores=4, random_seed=[1,2,3,4]) #draw 3000 posterior samples using NUTS sampling
            sim = pm3.sample_posterior_predictive(trace, samples=1000)
    
        return trace, sim
    
    def glm_log10(x_, y_, maxConc):
        x=np.array(x_).copy()
        y=np.array(y_).copy()
        logger.info("x_values = {}".format(x))
        logger.info("y_values = {}".format(y))
        logger.info("maxConc = {}".format(maxConc))
        with pm3.Model() as model1:
            sigma = pm3.HalfCauchy('sigma', beta=1e7)
            cumSurv = pm3.Uniform('cumSurv', 1.e-6, maxConc,testval=0.7)
            epsilon = pm3.Normal('epsilon',0, tau=1./1e8**2,shape=x.shape[0]) 
            
            mu = np.log10((x+epsilon)*cumSurv) 
            yprime = np.log10(y)
            liklelihood = pm3.Normal('log10y',mu=mu, tau=1./sigma**2, observed=yprime)
    
            logger.info("Sampling for inference")
            trace  = pm3.sample(1000, tune=8000, cores=4, random_seed=[1,2,3,4]) 
            sim = pm3.sample_posterior_predictive(trace, samples=1000)
    
        return trace, sim

    if regselect=="Linear":
        logger.info("mixed effect regression in linear space; inferring cumSurv as the slope")
        logger.info("mixed effect model in Bayesian framework for AUC=B0*cumSurv")
        initialInnocCol_ind = refPAPs.columns.isin([0,'0'])

        x_ = refPAPs.loc[:,initialInnocCol_ind].values.flatten()
        y_ = refPAPs['aucFull'].values

        trace, sim = glm_linear(x_, y_)
        return trace, sim


    elif regselect=="Log10":
        logger.info("mixed effect regression in log10 space; inferring cumSurv from the intercept")
        logger.info("mixed effect model in Bayesian framework for log10AUC=log10B0+log10cumSurv")
        initialInnocCol_ind = refPAPs.columns.isin([0,'0'])

        x_ = refPAPs.loc[:,initialInnocCol_ind].values.flatten()
        y_ = refPAPs['aucFull'].values

        trace, sim = glm_log10(x_, y_, maxConc)
        return trace, sim 
        
    else:
        logger.error("invalid regselect; please select either Linear or Log10 regression")
        return None

def crossValidate(refTrace, refPAPs, maxConc, frac=0.8, niter=50):
    n_train = np.round(refPAPs.shape[0]*frac)
    n_valid = np.round(refPAPs.shape[0]*(1-frac))
    logger.info("{} PAPs will be used as reference and {} for detection over {} iterations".format(n_train, n_valid, niter))
    xv_CSP=[]
    q25=[]
    q975=[]
    n_train = []
    n_valid = []
    for i in range(niter):
        logger.info("----------XV iteration {} ------------".format(i))
        train = refPAPs.sample(frac=frac, replace=False, random_state=i)
        valid = refPAPs.loc[~refPAPs.index.isin(train.index),:]
        n_train.append(train.shape[0])
        n_valid.append(valid.shape[0])
        trace_train, sim_train = refStrainProfiling(train, 'Log10', maxConc)
        qs = np.array(mquantiles(trace_train['cumSurv'],[0.025,0.975]))
        q25.append(qs[0])
        q975.append(qs[1])
        valid_cumSurv = valid['aucFull'].values/valid['0'].values
        xv_CSP.append((valid_cumSurv>=qs[0]).sum())
    ntrain = np.mean(n_train)
    nvalid = np.mean(n_valid)
    perc_xv = np.mean(np.array(xv_CSP)/nvalid)
    std_xv = np.std(np.array(xv_CSP)/nvalid)
    q25_CV = np.std(q25)/np.mean(q25)
    q975_CV = np.std(q975)/np.mean(q975)
    logger.info("Saving a csv file crossValidation_CSP.csv for cross validation results")
    df_xv = pd.DataFrame({'n_train':n_train,'n_valid':n_valid,'n_xvalid':xv_CSP,'train_q2.5%':q25,'train_q97.5%':q975})
    df_xv.to_csv("crossValidation_CSP.csv", index=False)
    return  perc_xv, std_xv, q25_CV, q975_CV

def verifyModel(refTrace, simRef, refPAPs, regselect):
    initialInnocCol_ind = refPAPs.columns.isin([0,'0'])
    ref_initialInnoc = refPAPs.loc[:,initialInnocCol_ind].values.flatten()
    fig, ax=pl.subplots(figsize=(8,7))
    if regselect == 'Linear':
        qs = mquantiles(simRef['y'], [0.025,0.975],axis=0)
        y_sim = simRef['y'].mean(axis=0)
        ax.plot(np.log10(refPAPs['0']), np.log10(simRef['y'].T), '.', color='lightgray', linestyle="", alpha=0.05)
        ax.plot(np.log10(refPAPs['0']), np.log10(simRef['y'].T[:,0]), '.', markersize=20, color='lightgray', linestyle="", alpha=0.3, label='posterior predictive')
        sbs.scatterplot(data=np.log10(refPAPs[['0','aucFull']]), x='0', y='aucFull', s=100, label='data', ax=ax)
        sbs.lineplot(np.log10(refPAPs['0']),np.log10(refTrace['cumSurv'].mean()*refPAPs['0']), label="regression fit", ax=ax)
        ax.set_xlabel("log10(initial innoculum ($\mu$g/ml))")
        ax.set_ylabel("log10(PAP-AUC (CFU/ml.$\mu$g/ml))")
        ax.legend(facecolor='white', fontsize=14)

    elif regselect == 'Log10':
        qs = mquantiles(simRef['log10y'], [0.025,0.975],axis=0)
        y_sim = simRef['log10y'].mean(axis=0)
        ax.plot(np.log10(refPAPs['0']), simRef['log10y'].T, '.', color='lightgray', linestyle="", alpha=0.05)
        ax.plot(np.log10(refPAPs['0']), simRef['log10y'].T[:,0], '.', markersize=20, color='lightgray', linestyle="", alpha=0.3, label='posterior predictive')
        sbs.scatterplot(data=np.log10(refPAPs[['0','aucFull']]), x='0', y='aucFull', s=100, label='data', ax=ax)
        sbs.lineplot(np.log10(refPAPs['0']),np.log10(refTrace['cumSurv'].mean()*refPAPs['0']), label="regression fit", ax=ax)
        ax.set_xlabel("log10(initial innoculum ($\mu$g/ml))")
        ax.set_ylabel("log10(PAP-AUC (CFU/ml.$\mu$g/ml))")
        ax.legend(facecolor='white', fontsize=14)

    fig.savefig("regFit_CSP_"+regselect+".png")

def classifyIsos(refTrace, isoPAPs, regselect):
    if regselect=='Linear' or regselect=='Log10':
        initialInnocCol_ind = isoPAPs.columns.isin([0,'0'])
        threshold = np.array(mquantiles(refTrace['cumSurv'],[0.025]))[0]
        logger.info("{} regression: cumulative survival q2.5% threshold = {}".format(regselect,threshold))
        isoPAPs.loc[:,'class'+regselect+'Reg'] = (isoPAPs['aucFull'].values/isoPAPs.loc[:,initialInnocCol_ind].values.flatten() >=threshold).astype(float)
        return isoPAPs
    else:
        logger.error("there are only two choices for classifying isolates with CSP profiling of reference: 'Linear' or 'Log10'")
        return None


def main():
    args_parser = argparse.ArgumentParser(description='This program is built to analyze a set of PAPs to profile a reference strain (e.g. Mu3) and then use this profiling to predict heteroresistance in a set of clinical isolates')
    args_parser.add_argument('--refPAPdata','-m',help='a CSV file encompassing population analysis profile data for Mu3 with specific format: rows named by the isolates, columns named by the antimicrobial concentration (mic-g/mL) in ascending order starting from 0 and entries are the counts in (CFU/mL)', required=True)
    args_parser.add_argument('--IsoPAPdata','-i',help='a CSV file having population analysis profiling data for isolates with specific format: rows named by the isolates, columns named by the antimicrobial concentration (mic-g/mL) in ascending order starting from 0 and entries are the CFUs in (cells/mL)', required=False)
    args_parser.add_argument('--lowestDetectableCFU','-b',help='lowest detectable CFU that will be used to replace any fewer counts (default=0.0)', default=0.0) 
    args_parser.add_argument('--regselect','-s',help='select whether regression is in "Linear" or in "Log10" space (default)', default="Log10")
    args_parser.add_argument('--xvfrac','-xf',help='select fraction of reference PAPs for training set in cross-validation (default=0.8)', default=0.8)
    args_parser.add_argument('--xviter','-xi',help='select number of iterations for cross-validation (default=10)', default=10)

    args = args_parser.parse_args()
    lowestDetectableCFU = args.lowestDetectableCFU
    regselect = args.regselect
    xvfrac = float(args.xvfrac)
    xviter = int(args.xviter)
    if xvfrac>=1.0:
        logger.error("illegal fraction for splitting the data for the reference for cross-validation; use a fraction <1 (e.g. 0.8)")
        sys.exit(1)
    if xviter > 50:
        logger.warning("setting large number of iterations in the cross-validation will increase the computational time")
    if xviter < 20:
        logger.warning("setting small number of iterations in the cross-validation might not reflect the true accuracy")


    example=""" \
,0,1,2,3,4,6,8
Mu3_1,860000000.0,730000000.0,,,,170.0,40.0
Mu3_2,720000000.0,690000000.0,,,,340.0,10.0
Mu3_3,450000000.0,250000000.0,200000.0,35000.0,5000.0,350.0,0.0
Mu3_4,400000000.0,300000000.0,300000.0,29000.0,4000.0,980.0,
Mu3_5,270000000.0,360000000.0,400000.0,40000.0,,300.0,30.0
Mu3_6,270000000.0,460000000.0,600000.0,32000.0,,300.0,20.0
Mu3_6,9000000.0,7500000.0,,40.0,0.0,0.0,0.0
Mu3_7,18600000.0,14800000.0,201000.0,440.0,0.0,0.0,0.0
"""            

    logger.info("Reading Reference Data")
    refPAPs = pd.read_csv(args.refPAPdata, index_col=0)
    if ~np.any(refPAPs.columns.isin([0,'0'])):
        logger.error("the Mu3 data is not correctly formatted: Rows named by the isolates, columns named by the antimicrobial concentration (mic-g/mL) in ascending order starting from 0 and entries are the CFUs in (cells/mL); here is an example:")
        logger.info("{}".format(example))
        sys.exit(0)

    concs = refPAPs.columns.values
    maxAntibioticConc = concs.astype(float).max()
    logger.info("the concentration gradient is {}".format(concs.astype(float)))
    logger.info("the maximum concentration is {}".format(maxAntibioticConc))

    logger.info("Cleaning Reference Data")
    refPAPs = cleanData(dataF=refPAPs,lowestDetectableCFU=lowestDetectableCFU, minimumNumOfPoints=2)
    logger.info("Calculating AUC")
    refPAPs = addAUCs(inDF=refPAPs, concs=concs, uselog10=True) 
    refPAPs = addAUCs(inDF=refPAPs, concs=concs, uselog10=False) 
    logger.info("Running Reference-Strain-Profiling (CSP)")
    refTrace, simRef = refStrainProfiling(refPAPs, regselect=regselect, maxConc=maxAntibioticConc)
    df_trace = pm3.trace_to_dataframe(refTrace)
    logger.info("Saving MCMC trace to refTrace.csv")
    df_trace.to_csv("refTrace.csv", index=False)
    #plot 
    logger.info("Plotting model fit for visual inspection")
    verifyModel(refTrace, simRef, refPAPs, regselect=regselect)
    logger.info("Running Cross-Validation")
    perc_xv, std_xv, q25_CV, q975_CV = crossValidate(refTrace, refPAPs, maxConc=maxAntibioticConc, frac=xvfrac, niter=xviter)
    logger.info("Saving Cross-Validation Summary to crossValidation_CSP_summary.txt")
    xv_dict = {'mean accuracy': perc_xv, 'std accuracy':std_xv, 'q2.5%-CV':q25_CV, 'q97.5%-CV':q975_CV}
    f=open("crossValidation_CSP_summary.txt","wt")
    f.write( str(xv_dict) )
    f.close()
    logger.info("XV of CSP on reference: {}+/-{} detected".format(perc_xv,std_xv))
    logger.info("XV of CSP on reference: coefficient of variation for q2.5% = {}".format(q25_CV))
    logger.info("XV of CSP on reference: coefficient of variation for q97.5% = {}".format(q975_CV))

    if args.IsoPAPdata:
        logger.info("isolate data is provided will detect heteroresistance")
        logger.info("Reading Isolates' Data")
        isoPAPs = pd.read_csv(args.IsoPAPdata, index_col=0)
        isoPAPs = isoPAPs[concs]
        if ~np.any(isoPAPs.columns.isin([0,'0'])):
            logger.error("the isolates' data is not correctly formatted: Rows named by the isolates, columns named by the antimicrobial concentration (mic-g/mL) in ascending order starting from 0 and entries are the CFUs in (cells/mL); here is an example:")
            logger.info("{}".format(example))
            sys.exit(0)

        logger.info("Cleaning Isolates' Data")
        isoPAPs = cleanData(dataF=isoPAPs, lowestDetectableCFU=lowestDetectableCFU, minimumNumOfPoints=2)
        logger.info("Calculating AUC")
        isoPAPs = addAUCs(inDF=isoPAPs, concs=concs, uselog10=True) 
        isoPAPs = addAUCs(inDF=isoPAPs, concs=concs, uselog10=False) 
        logger.info("Classifying Isolates")
        isoPAPs = classifyIsos(refTrace=refTrace, isoPAPs=isoPAPs, regselect=regselect)
        output_df = refPAPs.append(isoPAPs)
        logger.info("Saving reference and appended isolates' PAPs to ref_and_isolates_paps.csv")
        output_df.to_csv("ref_and_isolates_paps.csv", index=False)
        return df_trace, output_df
    else:
        logger.info("Saving reference PAPs to ref_paps.csv")
        refPAPs.to_csv("ref_paps.csv", index=False)
        return df_trace, refPAPs


if __name__ == "__main__":
    output=main()
