#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lockie
"""

###############################################################################

# Sustainable Factor Investing: Where Doing Well Meets Doing Good #

###############################################################################

import pandas as pd
import csv
import numpy as np
import os
from functools import reduce
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

###############################################################################

# get and change to dir
os.getcwd()
os.chdir('//') #set to directory

#import data
df = pd.read_csv('paperdata.csv', index_col='date', parse_dates=True)
print(df.head())

df.sort_values(by=['grouptcode'])

#drops nan obs and duplicate observations
df = df.dropna(subset=['mktcap'])  

#creates a unique id for each grouptcode
df['id'] = df.groupby(['grouptcode']).ngroup()

#set id to index
df = df.set_index(['id','grouptcode'], append = True)

#set all pricerelative values = -9 now nan
df.loc[df['pricerelative'] == -9, 'pricerelative'] = np.nan

#create return variable
df['ret'] = df['pricerelative'] - 1

#create book-to-market variable for value sort
df['bm'] = 1/df['ptb']

#cumulative return by grouptcode
df['cumret'] = (df['ret'] + 1).groupby(level=['id']).cumprod()

#mom signal
df['mom'] = df.groupby(level = ['id'])['cumret'].apply(lambda x: x.pct_change(11).shift())
###############################################################################
###############################################################################

''' Overall, the results are similar for quality, momentum, minimum volatility
and value. Size produces inconsistent results with the paper. When sorting the 
portfolios, there is a Python specific error which makes duplicates = 'drop' to 
be included in the line of code.

There are some parts of the below code which I know can be improved but I am 
just not sure how to at the current moment. It will come as I keep using 
Python! '''

###############################################################################
###############################################################################

#### Table 1 Stock counts and ESG scores ####

###############################################################################
###############################################################################

#Count of All sample, mean return and median market-cap
T1_alldf = df.copy()
T1_alldf = T1_alldf[np.isfinite(T1_alldf['pricerelative'])]
T1_alldf['count'] = 1
totalcountall = (T1_alldf.groupby(['date', 'sector'])['count'].sum())['2006-01-01':]
medmktcapall = T1_alldf.groupby(['sector'])['mktcap'].median()
meanretall = T1_alldf.groupby(['sector'])['ret'].mean()


#Count of ESG rated sample, mean return, median market-cap and mean ESG score
T1_ESGdf = (df.copy()).dropna(subset = ['ESG'])
T1_ESGdf = T1_ESGdf[np.isfinite(T1_ESGdf['pricerelative'])]
T1_ESGdf['count'] = 1
totalcountESG = (T1_ESGdf.groupby(['date', 'sector'])['count'].sum())['2006-01-01':]
medmktcapESG = T1_ESGdf.groupby(['sector'])['mktcap'].median()
meanretESG = T1_ESGdf.groupby(['sector'])['ret'].mean()
meanESGscore = T1_ESGdf.groupby(['date','sector'])['ESG'].mean()

###############################################################################
###############################################################################

#### Table 2 ESG integrated factors ####

###############################################################################
###############################################################################

#Standard factors before integration of ESG
result = [] # this list will store results
columns = ['roe','sd','bm','mom','mktcap'] # factors

for col in columns:
    data = df.copy()
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    if col == 'mom':
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on mom factor
    else:
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x,
            4, labels = False)) # sort on all other factors

    # value-weighted portoflio returns
    data = data.set_index(['port'], append = True)
    data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
    data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    
    #reshape long to wide
    data = data.groupby(['date', 'port'])['w_ret'].sum()
    data = data['2006-01-01':]
    data = data.unstack()
    if col == 'sd' or col == 'mktcap':
        data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
    else:
       data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
    data = data.stack().reset_index().set_index(['date'])
    data['port'] = data['port'].map({0.0:col+'0',
                     1.0:col+'1',
                     2.0:col+'2',
                     3.0:col+'3',
                     4.0:col+'LS'})
    data = data.reset_index().set_index(['date', 'port']).unstack()

    result.append(data) #this store the result in position 0 then 1 then 2 etc

#join factor returns together 
standard_factors = [result[0], result[1], result[2], result[3], result[4]]
standard_factors = standard_factors[0].join(standard_factors[1:])

###############################################################################
###############################################################################

#All returns
dfall = df.copy()
dfall['mktcap'] = dfall.groupby(level = ['id']).mktcap.shift(1)

#VW portoflio returns
dfall['tmktcap'] = dfall.groupby(['date'])['mktcap'].transform(sum)
dfall['w_ret'] = (dfall.mktcap / dfall.tmktcap) * dfall.ret

#reshape long to wide
dfall = dfall.groupby(['date'])['w_ret'].sum()
dfall = dfall['2006-01-01':].rename('all')

###############################################################################

#ESG rated portfolio returns
dfrated = df.copy()
dfrated = dfrated.dropna(subset = ['ESG']) # drop if no ESG score
dfrated['mktcap'] = dfrated.groupby(level = ['id']).mktcap.shift(1)

# value-weighted portoflio returns
dfrated['tmktcap'] = dfrated.groupby(['date'])['mktcap'].transform(sum)
dfrated['w_ret'] = (dfrated.mktcap / dfrated.tmktcap) * dfrated.ret

#reshape long to wide
dfrated = dfrated.groupby(['date'])['w_ret'].sum()
dfrated = dfrated['2006-01-01':].rename('rated')

###############################################################################
###############################################################################

#Method 1 - ESG integration through negative screening - drop all Non-ESG firms
M1result = [] # this list will store results
columns = ['roe','sd','bm','mom','mktcap'] # factors

for col in columns:
    data = df.copy()
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on factors

    # value-weighted portoflio returns
    data = data.set_index(['port'], append = True)
    data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
    data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    data['w_ret'] = data['w_ret']
    
    #reshape long to wide
    data = data.groupby(['date', 'port'])['w_ret'].sum()
    data = data['2006-01-01':]
    data = data.unstack()
    if col == 'sd' or col == 'mktcap':
        data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
    else:
       data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
    data = data.stack().reset_index().set_index(['date'])
    data['port'] = data['port'].map({0.0:'M1'+col+'0',
                     1.0:'M1'+col+'1',
                     2.0:'M1'+col+'2',
                     3.0:'M1'+col+'3',
                     4.0:'M1'+col+'LS'})
    data = data.reset_index().set_index(['date', 'port']).unstack()

    M1result.append(data) 

#join factor returns together 
M1ESG_factors = [M1result[0], M1result[1], M1result[2], M1result[3], M1result[4]]
M1ESG_factors = M1ESG_factors[0].join(M1ESG_factors[1:])

###############################################################################
###############################################################################

#Method 2 - ESG integration through negative screening after sorting portfolio - drop all Non-ESG firms
M2result = [] # this list will store results
columns = ['roe','sd','bm','mom','mktcap'] # factors

for col in columns:
    data = df.copy()
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on factors
    data = data.dropna(subset = ['ESG']) # drop if no ESG score

    # value-weighted portoflio returns
    data = data.set_index(['port'], append = True)
    data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
    data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    data['w_ret'] = data['w_ret']
    
    #reshape long to wide
    data = data.groupby(['date', 'port'])['w_ret'].sum()
    data = data['2006-01-01':]
    data = data.unstack()
    if col == 'sd' or col == 'mktcap':
        data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
    else:
       data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
    data = data.stack().reset_index().set_index(['date'])
    data['port'] = data['port'].map({0.0:'M2'+col+'0',
                     1.0:'M2'+col+'1',
                     2.0:'M2'+col+'2',
                     3.0:'M2'+col+'3',
                     4.0:'M2'+col+'LS'})
    data = data.reset_index().set_index(['date', 'port']).unstack()

    M2result.append(data)

#join factor returns together 
M2ESG_factors = [M2result[0], M2result[1], M2result[2], M2result[3], M2result[4]]
M2ESG_factors = M2ESG_factors[0].join(M2ESG_factors[1:])

###############################################################################
###############################################################################

#Method 3 - ESG interaction with sorting signal
M3result = [] # this list will store results
columns = ['roe','sd','bm','mom','mktcap'] # factors

for col in columns:
    data = df.copy().reset_index().set_index(['date'])
    data['cap'] = data.groupby(['id']).mktcap.shift()
    data = data.loc['2006-01-01':]
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data = data[np.isfinite(data[col])]
    data['rank'] = data[col].rank(method='first')
    
    #sort portfolio on signal decile
    data['port'] = data.groupby(['date'])['rank'].transform(lambda x: pd.qcut(x,
         10, labels = False)) # sort on factors
    
    #sort portfolio on bm decile by port
    data['port1'] = data.groupby(['date', 'port'])[col].transform(lambda x: pd.qcut(x,
         10, labels = False, duplicates = 'drop')) # sort on bm 
    
    data['port'] = data['port']*10
    data['port1'] = data['port'] + data['port1']
    data['port2'] = (0.5*data['ESG'] + 0.5*data['port1']) # interaction factor
    
    # sorting signal for interaction factor
    data['sort'] = data.groupby(['date'])['port2'].transform(lambda x: pd.qcut(x,
         4, labels = False, duplicates = 'drop' )) # sort on interaction factor
    
    # value-weighted portoflio returns
    data = data.set_index(['sort'], append = True)
    data['tmktcap'] = data.groupby(['date','sort'])['cap'].transform(sum)
    data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    
    #reshape long to wide
    data = data.groupby(['date', 'sort'])['w_ret'].sum()
    data = data.unstack()
    if col == 'sd' or col == 'mktcap':
        data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
    else:
       data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
    data = data.stack().reset_index().set_index(['date'])
    data['sort'] = data['sort'].map({0.0:'M3'+col+'0',
                     1.0:'M3'+col+'1',
                     2.0:'M3'+col+'2',
                     3.0:'M3'+col+'3',
                     4.0:'M3'+col+'LS'})
    data =  data.rename(columns={'sort': 'port'}).reset_index().set_index(['date', 'port']).unstack()
    
    M3result.append(data)

#join M3 factor returns together 
M3ESG_factors = [M3result[0], M3result[1], M3result[2], M3result[3], M3result[4]]
M3ESG_factors = M3ESG_factors[0].join(M3ESG_factors[1:])

###############################################################################
###############################################################################

#merge all portfolio returns together
df_returns = [standard_factors,M1ESG_factors,M2ESG_factors,M3ESG_factors]
returns = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), df_returns)

#merge with riskfree, risk premium factor
dfMKT = df.copy()
dfMKT['mktcap'] = dfMKT.groupby(level = ['id']).mktcap.shift(1)

#total market cap by each date and port			
dfMKT['tmktcap'] = dfMKT.groupby(['date'])['mktcap'].transform(sum)

#VW return
dfMKT['w_ret'] = ((dfMKT.mktcap / dfMKT.tmktcap) * dfMKT.ret)
dfMKT['rf'] = dfMKT['rfprel'] - 1
dfMKT = dfMKT.set_index(dfMKT['rf'],append = True).groupby(['date','rf'])['w_ret'].sum().reset_index(level=[1]).rename(columns={'w_ret': 'rm'})
dfMKT['rm'] = dfMKT['rm'].shift(-1)
dfMKT['rp'] = dfMKT.rm- dfMKT.rf
MKT = dfMKT['2006-01-01':]

df_all_rated = [dfall, dfrated, MKT]
returns = returns.join(df_all_rated)
returns.columns=returns.columns.str[1]
returns = returns.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

###############################################################################
###############################################################################

#Performance statistics
def maxdd(returns):  
    r = returns.add(1).cumprod()
    dd = r.div(r.cummax()) - 1
    mdd = dd.min()
    mdd = pd.DataFrame(mdd, columns = ['MaxDD'])
    return mdd.T
    
def skewandkurt(returns):
    skewness = returns.skew()
    sk = pd.DataFrame(skewness, columns = ['Skewness'])
    kurtosis = returns.kurtosis()
    kurt = pd.DataFrame(kurtosis, columns = ['Kurtosis'])
    skew_kurt = sk.join(kurt)
    return skew_kurt.T
    
def sharpe(returns):
    monthly_return = returns.mean()
    mret = pd.DataFrame(monthly_return, columns = ['Monthly Return'])
    monthly_volatility = returns.std()
    mvol = pd.DataFrame(monthly_volatility, columns = ['Monthly Volatility'])
    xrets = returns.sub(returns['rf'],axis = 0)
    xrets = xrets.mean()
    sharpe = (xrets/monthly_volatility)*np.sqrt(12)
    sharpe_ratio =pd.DataFrame(sharpe, columns = ['Sharpe'])
    dfs = [mret, mvol, sharpe_ratio]
    dfs = dfs[0].join(dfs[1:])
    return dfs.T
 
#t-statistics 

tstat = []
columns = returns.columns

for col in columns:
     t_statistic, p_value = stats.ttest_1samp(returns.dropna()[col], 0)
     tstat.append(t_statistic)

tstat_T2 = pd.DataFrame(tstat, index = columns).rename(columns={0: 't-statistic'}).T

###############################################################################

#merge performance statistics into one DataFrame
maxdd_T2 = maxdd(returns)
skew_T2 = skewandkurt(returns)
sharpe_T2 = sharpe(returns)

T2stats = [sharpe_T2,tstat_T2,skew_T2,maxdd_T2]
T2stats = (pd.concat(T2stats)).round(4)

T2stats.iloc[:,:-3].to_csv('T2stats.csv', sep = ',')

###############################################################################
###############################################################################

#### Table 4 Factor loadings ####

###############################################################################
###############################################################################

#FF factors
SMB = []
HML = []
UMD = []

FFcols = ['bm', 'mom']

for col in FFcols:
    data = df.copy()
    #first sort on size
    data['port1'] = data.groupby(['date'])['mktcap'].transform(lambda x: pd.qcut(x,
         2, labels = False, duplicates = 'drop'))
    if col == 'mom':
        #second sort on mom
        data['port2'] = data.groupby(['date','port1'])['mom'].transform(lambda x: pd.qcut(x,
     3, labels = False, duplicates ='drop'))
    else:
       #second sort on bm
       data['port2'] = data.groupby(['date','port1'])['bm'].transform(lambda x: pd.qcut(x, 
         3, labels = False, duplicates ='drop')) 
    data['mktcap'] = data.groupby(level = ['id']).mktcap.shift(1)
    data['tmktcap'] = data.groupby(['date','port1','port2'])['mktcap'].transform(sum)
    # VW return
    data['w_ret'] = ((data.mktcap / data.tmktcap) * data.ret)
    data = data.groupby(['date', 'port1','port2'])['w_ret'].sum()
    data = data['2006-01-01':]
    # SL SM SH BL BM BH
    data = data.reset_index().set_index(['date'])
    data['port1'] = data['port1'].map({0.0:'S',1.0:'B'})
    data['port2'] = data['port2'].map({0.0:'L',1.0:'M',2.0:'H'})
    S = data['port1'] + data['port2']
    data = data.set_index(S, append=True)['w_ret'].unstack()
    if col == 'bm':
        #SMB 
        data['SMB'] = (1/3*data['SL']+1/3*data['SM']+1/3*data['SH']) - (1/3*data['BL']+1/3*data['BM']+1/3*data['BH'])
        SMB.append(data['SMB'])
        #HML
        data['HML'] = (1/2*data['SH']+1/2*data['BH']) - (1/2*data['SL']+1/2*data['BL'])
        HML.append(data['HML'])
    else:
        #UMD
        data['UMD'] = (1/2*data['SH'] + 1/2*data['BH']) - (1/2*data['SL'] + 1/2*data['BL'])
        UMD.append(data['UMD'])

#SMB, HML and UMD DataFrame    
SMB = pd.DataFrame(SMB).stack().reset_index([0]).drop(['level_0'], 
                axis = 1).rename(columns={0: 'SMB'})

HML = pd.DataFrame(HML).stack().reset_index([0]).drop(['level_0'], 
                axis = 1).rename(columns={0: 'HML'})

UMD = pd.DataFrame(UMD).stack().reset_index([0]).drop(['level_0'], 
                axis = 1).rename(columns={0: 'UMD'})

#join regression factors 
FF_factors = [SMB, HML, UMD]
FF_factors = FF_factors[0].join(FF_factors[1:])

###############################################################################

#reg factors from paper
paperFF = pd.read_csv('Regressionfactorspaper.csv', index_col = 'date', parse_dates = True)
paperFF.index = FF_factors.index
fac_ret_paper = paperFF.join(returns)

# Can use these to test using the factors constructed through python. Only difference here,
# whole Australian sample is not used. If the factors constructed here are used in Stata, 
# they produce consistent results. This demonstrates that this works correctly.

#fac_ret_paper = FF_factors.join(returns)

###############################################################################

df1 = fac_ret_paper.copy()

#subtract risk free rate from portfolio strategies returns
columns = df1.iloc[:,4:-3].columns
for col in columns:
    df1[col] = (df1[col] - df1['rf'])
 
#size and min vol long short port1 - port4
col = ['mktcap','sd']
methods = ['M1', 'M2', 'M3']

for cols in col:
    for meths in methods:
           df1[meths+cols+'LS'] = (df1[meths+cols+'0'] - df1[meths+cols+'3'])

#quality, value and momentum long short port4 - port1
col = ['roe','bm','mom']
methods = ['M1', 'M2', 'M3']

for cols in col:
    for meths in methods:
        df1[meths+cols+'LS'] = (df1[meths+cols+'3'] - df1[meths+cols+'0'])
 
###############################################################################
       
## T-test each strategy     
ttest = []
columns = df1.iloc[:,4:-3].columns

for col in columns:
     t_statistic, p_value = stats.ttest_1samp(df1.dropna()[col], 0)
     ttest.append(t_statistic)

ttest = pd.DataFrame(ttest, index = columns).rename(columns={0: 't-statistic'}).T

###############################################################################
###############################################################################

#Regression      
modelcoeff = []
modeltvalues = [] 
modelrsquared = []

df1_reg = df1.iloc[:-1,]
columns = df1.iloc[:-1,4:-3].columns
fac = ['rp','SMB','HML','UMD']

for col in columns:
    data = df1_reg.copy()
    y_NaN = df1_reg[col].isnull().values.any()
    if y_NaN == True:
        X = df1_reg[fac]
        y = df1_reg.dropna()[col]
        df_reg = X.join(y).dropna()
        X = sm.add_constant(df_reg[fac])
        y = df_reg[col]
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)
        modelcoeff.append(model.params)
        modeltvalues.append(model.tvalues)
        modelrsquared.append(model.rsquared)
    else:
        X = sm.add_constant(df1_reg[fac])
        y = df1_reg[col]
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)
        modelcoeff.append(model.params)
        modeltvalues.append(model.tvalues)
        modelrsquared.append(model.rsquared)
 
modelcoeffcols = ({'const': 'const_coeff','rp': 'rp_coeff','SMB': 'SMB_coeff',
                  'HML': 'HML_coeff','UMD':'UMD_coeff'})
modeltvaluecols = ({'const': 'const_tstat','rp': 'rp_tstat','SMB': 'SMB_tstat',
                  'HML': 'HML_tstat','UMD':'UMD_tstat'})
modelrsquaredcols = ({0:'const_r2'})
            
modelcoeff = pd.DataFrame(modelcoeff, index = columns).rename(columns = modelcoeffcols)
modeltvalues = pd.DataFrame(modeltvalues, index = columns).rename(columns = modeltvaluecols)
modelrsquared = pd.DataFrame(modelrsquared, index = columns).rename(columns = modelrsquaredcols )

reg = [modelcoeff, modeltvalues, modelrsquared]
T3reg = (reg[0].join(reg[1:]).T).round(4)

T3reg.to_csv('T3reg.csv', sep = ',')

###############################################################################
###############################################################################     

#### Table 3 Business cycles and market conditions ####

###############################################################################
###############################################################################     

conditions = pd.read_csv('Conditions.csv', index_col='date', parse_dates=True)
conditions.index = returns[:-1].index
conditions_ret = conditions.join(returns)

#Expansion
exp = conditions_ret.loc[conditions_ret['cycle'] == 0]

#Recession
rec = conditions_ret.loc[conditions_ret['cycle'] == 1]

#High inflation
highinf = conditions_ret.loc[conditions_ret['inflation'] == 1]

#Low inflation
lowinf = conditions_ret.loc[conditions_ret['inflation'] == 0]

#High volatility
highvol = conditions_ret.loc[conditions_ret['vix'] == 1]

#Low volatility
lowvol = conditions_ret.loc[conditions_ret['vix'] == 0]

#High creditspread
highcs = conditions_ret.loc[conditions_ret['cs'] == 1]

#Low creditspread
lowcs = conditions_ret.loc[conditions_ret['cs'] == 0]

###############################################################################

#t-statistics 

results = {}
columns = conditions_ret.iloc[:,4:].columns
conds = [exp, rec, highinf, lowinf, highvol, lowvol, highcs, lowcs]
conds_names = ['exp_tstat', 'rec_tstat', 'highinf_tstat', 'lowinf_tstat', 'highvol_tstat', 'lowvol_tstat', 'highcs_tstat', 'lowcs_tstat']

i = -1
for cond in conds:
    i += 1
    name = conds_names[i]
    results[name] = [] # Initialise list
    for col in columns:
        t_statistic, p_value = stats.ttest_1samp(cond.dropna()[col], 0)
        results[name].append(t_statistic)

T4tstat = pd.DataFrame(results, index = columns).rename(columns={0: 't-statistic'}).T

#not sure exactly how to do this in a loop at the moment
sharpe_exp = (sharpe(exp)).rename(index = {'Monthly Return':'Monthly Return_exp','Monthly Volatility':'Monthly Volatility_exp','Sharpe':'Sharpe_exp'})
sharpe_rec = (sharpe(rec)).rename(index = {'Monthly Return':'Monthly Return_rec','Monthly Volatility':'Monthly Volatility_rec','Sharpe':'Sharpe_rec'})
sharpe_hinf = (sharpe(highinf)).rename(index = {'Monthly Return':'Monthly Return_hinf','Monthly Volatility':'Monthly Volatility_hinf','Sharpe':'Sharpe_hinf'})
sharpe_linf = (sharpe(lowinf)).rename(index = {'Monthly Return':'Monthly Return_linf','Monthly Volatility':'Monthly Volatility_linf','Sharpe':'Sharpe_linf'})
sharpe_hcs = (sharpe(highcs)).rename(index = {'Monthly Return':'Monthly Return_hcs','Monthly Volatility':'Monthly Volatility_hcs','Sharpe':'Sharpe_hcs'})
sharpe_lcs = (sharpe(lowcs)).rename(index = {'Monthly Return':'Monthly Return_lcs','Monthly Volatility':'Monthly Volatility_lcs','Sharpe':'Sharpe_lcs'})
sharpe_hvol = (sharpe(highvol)).rename(index = {'Monthly Return':'Monthly Return_hvol','Monthly Volatility':'Monthly Volatility_hvol','Sharpe':'Sharpe_hvol'})
sharpe_lvol = (sharpe(lowvol)).rename(index = {'Monthly Return':'Monthly Return_lvol','Monthly Volatility':'Monthly Volatility_lvol','Sharpe':'Sharpe_lvol'})

T4stats = [sharpe_exp,sharpe_rec,sharpe_hinf,sharpe_linf,sharpe_hcs,
           sharpe_lcs,sharpe_hvol,sharpe_lvol,T4tstat]
T4stats = (pd.concat(T4stats, sort=False)).round(4)

T4stats.iloc[:,4:-3].to_csv('T4stats.csv', sep = ',')

###############################################################################
###############################################################################

#### Table 5 Environment (E), Social (S) and Governance (G) integrated factors ####
 
###############################################################################
###############################################################################

#Method 1 - E, S, G integration through negative screening - drop all Non-E,S,G firms

M1resultEnv = [] 
M1resultSoc = []
M1resultGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['Env', 'Soc', 'Gov']

for col in columns:
    for score in disc:
        data = df.copy()
        data = data.dropna(subset = [score]) # drop if no score
        data['cap'] = data.groupby(level = ['id']).mktcap.shift()
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on factors

        # value-weighted portoflio returns
        data = data.set_index(['port'], append = True)
        data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
        data['w_ret'] = data['w_ret']
    
        #reshape long to wide
        data = data.groupby(['date', 'port'])['w_ret'].sum().shift(-4)
        data = data['2006-01-01':]
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
                data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
                data = data.stack().reset_index().set_index(['date'])
                data['port'] = data['port'].map({0.0:'M1'+col+'0',
                    1.0:'M1'+col+'1',
                    2.0:'M1'+col+'2',
                    3.0:'M1'+col+'3',
                    4.0:'M1'+col+'LS'})
                data = data.reset_index().set_index(['date', 'port']).unstack()
        else:
                data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
                data = data.stack().reset_index().set_index(['date'])
                data['port'] = data['port'].map({0.0:'M1'+col+'0',
                    1.0:'M1'+col+'1',
                    2.0:'M1'+col+'2',
                    3.0:'M1'+col+'3',
                    4.0:'M1'+col+'LS'})
                data = data.reset_index().set_index(['date', 'port']).unstack()
        if score == 'Env':
            M1resultEnv.append(data) 
        elif score == 'Soc':
            M1resultSoc.append(data)
        else:
            M1resultGov.append(data)
                    
#join factor returns together 

M1E_factors = [M1resultEnv[0], M1resultEnv[1], M1resultEnv[2], M1resultEnv[3], M1resultEnv[4]]
M1E_factors = M1E_factors[0].join(M1E_factors[1:])

M1S_factors = [M1resultSoc[0], M1resultSoc[1], M1resultSoc[2], M1resultSoc[3], M1resultSoc[4]]
M1S_factors = M1S_factors[0].join(M1S_factors[1:])

M1G_factors = [M1resultGov[0], M1resultGov[1], M1resultGov[2], M1resultGov[3], M1resultGov[4]]
M1G_factors = M1G_factors[0].join(M1G_factors[1:])

###############################################################################
###############################################################################

#Method 2 - E,S,G integration through negative screening after sorting portfolio - drop all Non-E,S,G firms

M2resultEnv = [] 
M2resultSoc = []
M2resultGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['Env', 'Soc', 'Gov']

for col in columns:
    for score in disc:
        data = df.copy()
        data['cap'] = data.groupby(level = ['id']).mktcap.shift()
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on factors
        data = data.dropna(subset = [score]) # drop if no score

        # value-weighted portoflio returns
        data = data.set_index(['port'], append = True)
        data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
        data['w_ret'] = data['w_ret']
    
        #reshape long to wide
        data = data.groupby(['date', 'port'])['w_ret'].sum().shift(-4)
        data = data['2006-01-01':]
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
            data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
            data = data.stack().reset_index().set_index(['date'])
            data['port'] = data['port'].map({0.0:'M2'+col+'0',
                1.0:'M2'+col+'1',
                2.0:'M2'+col+'2',
                3.0:'M2'+col+'3',
                4.0:'M2'+col+'LS'})
            data = data.reset_index().set_index(['date', 'port']).unstack()
        else:
            data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
            data = data.stack().reset_index().set_index(['date'])
            data['port'] = data['port'].map({0.0:'M2'+col+'0',
                1.0:'M2'+col+'1',
                2.0:'M2'+col+'2',
                3.0:'M2'+col+'3',
                4.0:'M2'+col+'LS'})
            data = data.reset_index().set_index(['date', 'port']).unstack()
        if score == 'Env':
            M2resultEnv.append(data) 
        elif score == 'Soc':
            M2resultSoc.append(data)
        else:
            M2resultGov.append(data)

#join factor returns together 

M2E_factors = [M2resultEnv[0], M2resultEnv[1], M2resultEnv[2], M2resultEnv[3], M2resultEnv[4]]
M2E_factors = M2E_factors[0].join(M2E_factors[1:])

M2S_factors = [M2resultSoc[0], M2resultSoc[1], M2resultSoc[2], M2resultSoc[3], M2resultSoc[4]]
M2S_factors = M2resultSoc[0].join(M2resultSoc[1:])

M2G_factors = [M2resultGov[0], M2resultGov[1], M2resultGov[2], M2resultGov[3], M2resultGov[4]]
M2G_factors = M2G_factors[0].join(M2G_factors[1:])

###############################################################################
###############################################################################

#Method 3 - E,S,G interaction with sorting signal

M3resultEnv = [] 
M3resultSoc = []
M3resultGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['Env', 'Soc', 'Gov']

for col in columns:
    for score in disc:
        data = df.copy().reset_index().set_index(['date'])
        data['cap'] = data.groupby(['id']).mktcap.shift()
        data = data.loc['2006-01-01':]
        data = data.dropna(subset = [score]) # drop if no score
        data = data[np.isfinite(data[col])]
        data['rank'] = data[col].rank(method='first')
        
        #sort portfolio on signal decile
        data['port'] = data.groupby(['date'])['rank'].transform(lambda x: pd.qcut(x,
            10, labels = False)) # sort on factors
        
        #sort portfolio on bm decile by port
        data['port1'] = data.groupby(['date', 'port'])[col].transform(lambda x: pd.qcut(x,
            10, labels = False, duplicates = 'drop')) # sort on bm 
    
        data['port'] = data['port']*10
        data['port1'] = data['port'] + data['port1']
        data['port2'] = (0.5*data[score] + 0.5*data['port1']) # interaction factor
    
        # sorting signal for interaction factor
        data['sort'] = data.groupby(['date'])['port2'].transform(lambda x: pd.qcut(x,
            4, labels = False, duplicates = 'drop' )) # sort on interaction factor
    
        # value-weighted portoflio returns
        data = data.set_index(['sort'], append = True)
        data['tmktcap'] = data.groupby(['date','sort'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    
        #reshape long to wide
        data = data.groupby(['date', 'sort'])['w_ret'].sum().shift(-4)
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
            data[4.0] = data[0.0] - data[3.0] # long(0.0) and short(3.0)
            data = data.stack().reset_index().set_index(['date'])
            data['sort'] = data['sort'].map({0.0:'M3'+col+'0',
                         1.0:'M3'+col+'1',
                         2.0:'M3'+col+'2',
                         3.0:'M3'+col+'3',
                         4.0:'M3'+col+'LS'})
            data =  data.rename(columns={'sort': 'port'}).reset_index().set_index(['date', 'port']).unstack()
        else:
            data[4.0] = data[3.0] - data[0.0] # long(3.0) and short(0.0)
            data = data.stack().reset_index().set_index(['date'])
            data['sort'] = data['sort'].map({0.0:'M3'+col+'0',
                         1.0:'M3'+col+'1',
                         2.0:'M3'+col+'2',
                         3.0:'M3'+col+'3',
                         4.0:'M3'+col+'LS'})
            data =  data.rename(columns={'sort': 'port'}).reset_index().set_index(['date', 'port']).unstack()
        if score == 'Env':
            M3resultEnv.append(data) 
        elif score == 'Soc':
            M3resultSoc.append(data)
        else:
            M3resultGov.append(data)
 
#join factor returns together 
M3E_factors = [M3resultEnv[0], M3resultEnv[1], M3resultEnv[2], M3resultEnv[3], M3resultEnv[4]]
M3E_factors = M3E_factors[0].join(M3E_factors[1:])

M3S_factors = [M3resultSoc[0], M3resultSoc[1], M3resultSoc[2], M3resultSoc[3], M3resultSoc[4]]
M3S_factors = M3S_factors[0].join(M3S_factors[1:])

M3G_factors = [M3resultGov[0], M3resultGov[1], M3resultGov[2], M3resultGov[3], M3resultGov[4]]
M3G_factors = M3G_factors[0].join(M3G_factors[1:])       
    
###############################################################################
###############################################################################

#merge all portfolio returns together

#Environment returns
E_df_returns = [M1E_factors,M2E_factors,M3E_factors]
ENVrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), E_df_returns)
ENVrets = ENVrets.join(df_all_rated)
ENVrets.columns=ENVrets.columns.str[1]
ENVrets = ENVrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

#Social returns
S_df_returns = [M1S_factors,M2S_factors,M3S_factors]
SOCrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), S_df_returns)
SOCrets = SOCrets.join(df_all_rated)
SOCrets.columns=SOCrets.columns.str[1]
SOCrets = SOCrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

#Governance returns
G_df_returns = [M1G_factors,M2G_factors,M3G_factors]
GOVrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), G_df_returns)
GOVrets = GOVrets.join(df_all_rated)
GOVrets.columns=GOVrets.columns.str[1]
GOVrets = GOVrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})
 
###############################################################################
###############################################################################

#Sharpe 
sharpe_Env = (sharpe(ENVrets)).rename(index = {'Monthly Return':'Monthly Return_ENV','Monthly Volatility':'Monthly Volatility_ENV','Sharpe':'Sharpe_ENV'})
sharpe_Soc = (sharpe(SOCrets)).rename(index = {'Monthly Return':'Monthly Return_SOC','Monthly Volatility':'Monthly Volatility_SOC','Sharpe':'Sharpe_SOC'})
sharpe_Gov = (sharpe(GOVrets)).rename(index = {'Monthly Return':'Monthly Return_GOV','Monthly Volatility':'Monthly Volatility_GOV','Sharpe':'Sharpe_GOV'})

# t-statistics 

results = {}
columns = ENVrets.columns
T5rets = [ENVrets, SOCrets, GOVrets] 
T5rets_names = ['ENVrets_tstat', 'SOCrets_tstat', 'GOVrets_tstat']

i = -1
for ret in T5rets:
    i += 1
    name = T5rets_names[i]
    results[name] = [] # Initialise list
    for col in columns:
        t_statistic, p_value = stats.ttest_1samp(ret.dropna()[col], 0)
        results[name].append(t_statistic)

T5tstat = pd.DataFrame(results, index = columns).rename(columns={0: 't-statistic'}).T

T5stats = [sharpe_Env,sharpe_Soc,sharpe_Gov,T5tstat]

T5stats = (pd.concat(T5stats)).round(4)

T5stats.iloc[:,:-3].to_csv('T5stats.csv', sep = ',')

###############################################################################
###############################################################################

#### Table 6 Best-of-sector (BOS) portfolios ####

###############################################################################
###############################################################################

#Method 1 - BOS E, S, G integration through negative screening - drop all Non-E,S,G firms

M1resultBOSESG = [] 
M1resultBOSEnv = [] 
M1resultBOSSoc = []
M1resultBOSGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['ESG', 'Env', 'Soc', 'Gov']

df.columns
for col in columns:
    for score in disc:
        data = df.copy()
        data = data.dropna(subset = [score]) # drop if no score
        data['cap'] = data.groupby(level = ['id']).mktcap.shift()
        data['port'] = data.groupby(['date','sector'])[col].transform(lambda x: pd.qcut(x, 
            3, labels = False, duplicates = 'drop')) # sort on factors - tercile and by sector

        # value-weighted portoflio returns
        data = data.set_index(['port'], append = True)
        data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
        data['w_ret'] = data['w_ret']
    
        #reshape long to wide
        data = data.groupby(['date', 'port'])['w_ret'].sum().shift(-3)
        data = data['2006-01-01':]
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
                data[3.0] = data[0.0] - data[2.0] # long(0.0) and short(2.0)
                data = data.stack().reset_index().set_index(['date'])
                data['port'] = data['port'].map({0.0:'M1'+col+'0',
                    1.0:'M1'+col+'1',
                    2.0:'M1'+col+'2',
                    3.0:'M1'+col+'LS'})
                data = data.reset_index().set_index(['date', 'port']).unstack()
        else:
                data[3.0] = data[2.0] - data[0.0] # long(2.0) and short(0.0)
                data = data.stack().reset_index().set_index(['date'])
                data['port'] = data['port'].map({0.0:'M1'+col+'0',
                    1.0:'M1'+col+'1',
                    2.0:'M1'+col+'2',
                    3.0:'M1'+col+'LS'})
                data = data.reset_index().set_index(['date', 'port']).unstack()
        if score == 'ESG':
            M1resultBOSESG.append(data) 
        elif score == 'Env':
            M1resultBOSEnv.append(data)
        elif score == 'Soc':
            M1resultBOSSoc.append(data)
        else:
            M1resultBOSGov.append(data)
                    
#join factor returns together 

M1BOSESG_factors = [M1resultBOSESG[0], M1resultBOSESG[1], M1resultBOSESG[2], M1resultBOSESG[3], M1resultBOSESG[4]]
M1BOSESG_factors = M1BOSESG_factors[0].join(M1BOSESG_factors[1:])

M1BOSE_factors = [M1resultBOSEnv[0], M1resultBOSEnv[1], M1resultBOSEnv[2], M1resultBOSEnv[3], M1resultBOSEnv[4]]
M1BOSE_factors = M1BOSE_factors[0].join(M1BOSE_factors[1:])

M1BOSS_factors = [M1resultBOSSoc[0], M1resultBOSSoc[1], M1resultBOSSoc[2], M1resultBOSSoc[3], M1resultBOSSoc[4]]
M1BOSS_factors = M1BOSS_factors[0].join(M1BOSS_factors[1:])

M1BOSG_factors = [M1resultBOSGov[0], M1resultBOSGov[1], M1resultBOSGov[2], M1resultBOSGov[3], M1resultBOSGov[4]]
M1BOSG_factors = M1BOSG_factors[0].join(M1BOSG_factors[1:])

###############################################################################
###############################################################################

#Method 2 - BOS E,S,G integration through negative screening after sorting portfolio 

M2resultBOSESG = [] 
M2resultBOSEnv = [] 
M2resultBOSSoc = []
M2resultBOSGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['ESG', 'Env', 'Soc', 'Gov']

for col in columns:
    for score in disc:
        data = df.copy()
        data['cap'] = data.groupby(level = ['id']).mktcap.shift()
        data['port'] = data.groupby(['date','sector'])[col].transform(lambda x: pd.qcut(x, 
            3, labels = False, duplicates = 'drop')) # sort on factors - tercile and by sector
        data = data.dropna(subset = [score]) # drop if no score

        # value-weighted portoflio returns
        data = data.set_index(['port'], append = True)
        data['tmktcap'] = data.groupby(['date','port'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
        data['w_ret'] = data['w_ret']
    
        #reshape long to wide
        data = data.groupby(['date', 'port'])['w_ret'].sum().shift(-3)
        data = data['2006-01-01':]
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
            data[3.0] = data[0.0] - data[2.0] # long(0.0) and short(3.0)
            data = data.stack().reset_index().set_index(['date'])
            data['port'] = data['port'].map({0.0:'M2'+col+'0',
                1.0:'M2'+col+'1',
                2.0:'M2'+col+'2',
                3.0:'M2'+col+'LS'})
            data = data.reset_index().set_index(['date', 'port']).unstack()
        else:
            data[3.0] = data[2.0] - data[0.0] # long(3.0) and short(0.0)
            data = data.stack().reset_index().set_index(['date'])
            data['port'] = data['port'].map({0.0:'M2'+col+'0',
                1.0:'M2'+col+'1',
                2.0:'M2'+col+'2',
                3.0:'M2'+col+'LS'})
            data = data.reset_index().set_index(['date', 'port']).unstack()
        if score == 'ESG':
            M2resultBOSESG.append(data)
        elif score == 'Env':
            M2resultBOSEnv.append(data)
        elif score == 'Soc':
            M2resultBOSSoc.append(data)
        else:
            M2resultBOSGov.append(data)

#join factor returns together 
M2BOSESG_factors = [M2resultBOSESG[0], M2resultBOSESG[1], M2resultBOSESG[2], M2resultBOSESG[3], M2resultBOSESG[4]]
M2BOSESG_factors = M2BOSESG_factors[0].join(M2BOSESG_factors[1:])

M2BOSE_factors = [M2resultBOSEnv[0], M2resultBOSEnv[1], M2resultBOSEnv[2], M2resultBOSEnv[3], M2resultBOSEnv[4]]
M2BOSE_factors = M2BOSE_factors[0].join(M2BOSE_factors[1:])

M2BOSS_factors = [M2resultBOSSoc[0], M2resultBOSSoc[1], M2resultBOSSoc[2], M2resultBOSSoc[3], M2resultBOSSoc[4]]
M2BOSS_factors = M2BOSS_factors[0].join(M2BOSS_factors[1:])

M2BOSG_factors = [M2resultBOSGov[0], M2resultBOSGov[1], M2resultBOSGov[2], M2resultBOSGov[3], M2resultBOSGov[4]]
M2BOSG_factors = M2BOSG_factors[0].join(M2BOSG_factors[1:])

###############################################################################
###############################################################################

#Method 3 - BOS E,S,G interaction with sorting signal

M3resultBOSESG = [] 
M3resultBOSEnv = [] 
M3resultBOSSoc = []
M3resultBOSGov = []

columns = ['roe','sd','bm','mom','mktcap'] # factors
disc = ['ESG', 'Env', 'Soc', 'Gov']

for col in columns:
    for score in disc:
        data = df.copy().reset_index().set_index(['date'])
        data['cap'] = data.groupby(['id']).mktcap.shift()
        data = data.loc['2006-01-01':]
        data = data.dropna(subset = [score]) # drop if no score
        data = data[np.isfinite(data[col])]
        data['rank'] = data[col].rank(method='first')
        
        #sort portfolio on signal decile
        data['port'] = data.groupby(['date','sector'])['rank'].transform(lambda x: pd.qcut(x,
            10, labels = False, duplicates = 'drop')) # sort on factors
        data = data.dropna(subset = ['port']) # drop if no score
        #sort portfolio on bm decile by port
        data['port1'] = data.groupby(['date', 'port'])['rank'].transform(lambda x: pd.qcut(x,
            10, labels = False, duplicates = 'drop')) # sort on bm 
    
        data['port'] = data['port']*10
        data['port1'] = data['port'] + data['port1']
        data['port2'] = (0.5*data[score] + 0.5*data['port1']) # interaction factor
    
        # sorting signal for interaction factor
        data['sort'] = data.groupby(['date'])['port2'].transform(lambda x: pd.qcut(x,
            3, labels = False, duplicates = 'drop')) # sort on interaction factor - tercile and by sector
    
        # value-weighted portoflio returns
        data = data.set_index(['sort'], append = True)
        data['tmktcap'] = data.groupby(['date','sort'])['cap'].transform(sum)
        data['w_ret'] = (data.cap / data.tmktcap) * data.ret
    
        #reshape long to wide
        data = data.groupby(['date', 'sort'])['w_ret'].sum().shift(-3)
        data = data.unstack()
        if col == 'sd' or col == 'mktcap':
            data[3.0] = data[0.0] - data[2.0] # long(0.0) and short(3.0)
            data = data.stack().reset_index().set_index(['date'])
            data['sort'] = data['sort'].map({0.0:'M3'+col+'0',
                1.0:'M3'+col+'1',
                2.0:'M3'+col+'2',
                3.0:'M3'+col+'LS'})
            data = data.reset_index().set_index(['date', 'sort']).unstack()
        else:
            data[3.0] = data[2.0] - data[0.0] # long(3.0) and short(0.0)
            data = data.stack().reset_index().set_index(['date'])
            data['sort'] = data['sort'].map({0.0:'M3'+col+'0',
                1.0:'M3'+col+'1',
                2.0:'M3'+col+'2',
                3.0:'M3'+col+'LS'})
            data = data.reset_index().set_index(['date', 'sort']).unstack()
        if score == 'ESG':
            M3resultBOSESG.append(data) 
        elif score == 'Env':
            M3resultBOSEnv.append(data) 
        elif score == 'Soc':
            M3resultBOSSoc.append(data)
        else:
            M3resultBOSGov.append(data)
 
#join factor returns together 
M3BOSESG_factors = [M3resultBOSESG[0], M3resultBOSESG[1], M3resultBOSESG[2], M3resultBOSESG[3], M3resultBOSESG[4]]
M3BOSESG_factors = M3BOSESG_factors[0].join(M3BOSESG_factors[1:])

M3BOSE_factors = [M3resultBOSEnv[0], M3resultBOSEnv[1], M3resultBOSEnv[2], M3resultBOSEnv[3], M3resultBOSEnv[4]]
M3BOSE_factors = M3BOSE_factors[0].join(M3BOSE_factors[1:])

M3BOSS_factors = [M3resultBOSSoc[0], M3resultBOSSoc[1], M3resultBOSSoc[2], M3resultBOSSoc[3], M3resultBOSSoc[4]]
M3BOSS_factors = M3BOSS_factors[0].join(M3BOSS_factors[1:])

M3BOSG_factors = [M3resultBOSGov[0], M3resultBOSGov[1], M3resultBOSGov[2], M3resultBOSGov[3], M3resultBOSGov[4]]
M3BOSG_factors = M3BOSG_factors[0].join(M3BOSG_factors[1:])    

###############################################################################

#merge all portfolio returns together (I am sure this can be done much easier, just not sure at the moment how)

#ESG returns
ESGBOS_df_returns = [M1BOSESG_factors,M2BOSESG_factors,M3BOSESG_factors]
BOSESGrets = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), ESGBOS_df_returns)
BOSESGrets = BOSESGrets.join(df_all_rated)
BOSESGrets.columns=BOSESGrets.columns.str[1]
BOSESGrets = BOSESGrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

#Environment returns
EBOS_df_returns = [M1BOSE_factors,M2BOSE_factors,M3BOSE_factors]
BOSENVrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), EBOS_df_returns)
BOSENVrets = BOSENVrets.join(df_all_rated)
BOSENVrets.columns=BOSENVrets.columns.str[1]
BOSENVrets = BOSENVrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

#Social returns
SBOS_df_returns = [M1BOSS_factors,M2BOSS_factors,M3BOSS_factors]
BOSSOCrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), SBOS_df_returns)
BOSSOCrets = BOSSOCrets.join(df_all_rated)
BOSSOCrets.columns=BOSSOCrets.columns.str[1]
BOSSOCrets = BOSSOCrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

#Governance returns
GBOS_df_returns = [M1BOSG_factors,M2BOSG_factors,M3BOSG_factors]
BOSGOVrets= reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                                how='outer'), GBOS_df_returns)
BOSGOVrets = BOSGOVrets.join(df_all_rated)
BOSGOVrets.columns=BOSGOVrets.columns.str[1]
BOSGOVrets = BOSGOVrets.rename(columns={'l':'all', 'a':'rated', 'f':'rf','m':'rm','p':'rp'})

###############################################################################
###############################################################################

#performance stats (I am sure again this could be improved)
sharpe_BOSESG = (sharpe(BOSESGrets)).rename(index = {'Monthly Return':'Monthly Return_BOSESG','Monthly Volatility':'Monthly Volatility_BOSESG','Sharpe':'Sharpe_BOSESG'})
sharpe_BOSEnv = (sharpe(BOSENVrets)).rename(index = {'Monthly Return':'Monthly Return_BOSEnv','Monthly Volatility':'Monthly Volatility_BOSEnv','Sharpe':'Sharpe_BOSEnv'})
sharpe_BOSSoc = (sharpe(BOSSOCrets)).rename(index = {'Monthly Return':'Monthly Return_BOSSoc','Monthly Volatility':'Monthly Volatility_BOSSoc','Sharpe':'Sharpe_BOSSoc'})
sharpe_BOSGov = (sharpe(BOSGOVrets)).rename(index = {'Monthly Return':'Monthly Return_BOSGov','Monthly Volatility':'Monthly Volatility_BOSGov','Sharpe':'Sharpe_BOSGov'})

sk_BOSESG = (skewandkurt(BOSESGrets)).rename(index = {'Skewness':'Skewness_BOSESG','Kurtosis':'Kurtosis_BOSESG'})
sk_BOSEnv = (skewandkurt(BOSENVrets)).rename(index = {'Skewness':'Skewness_BOSEnv','Kurtosis':'Kurtosis_BOSEnv'})
sk_BOSSoc = (skewandkurt(BOSSOCrets)).rename(index = {'Skewness':'Skewness_BOSSoc','Kurtosis':'Kurtosis_BOSSoc'})
sk_BOSGov = (skewandkurt(BOSGOVrets)).rename(index = {'Skewness':'Skewness_BOSGov','Kurtosis':'Kurtosis_BOSGov'})

maxdd_BOSESG = (maxdd(BOSESGrets)).rename(index = {'MaxDD':'MaxDD_BOSESG'})
maxdd_BOSEnv = (maxdd(BOSENVrets)).rename(index = {'MaxDD':'MaxDD_BOSENV'})
maxdd_BOSSoc = (maxdd(BOSSOCrets)).rename(index = {'MaxDD':'MaxDD_BOSSOC'})
maxdd_BOSGov = (maxdd(BOSGOVrets)).rename(index = {'MaxDD':'MaxDD_BOSGOV'})

#t-statistics 

results = {}
columns = BOSENVrets.columns
T6rets = [BOSESGrets, BOSENVrets, BOSSOCrets, BOSGOVrets]
T6rets_names = ['BOSESGrets_tstat', 'BOSENVrets_tstat', 'BOSSOCrets_tstat', 'BOSGOVrets_tstat']

i = -1
for ret in T6rets:
    i += 1
    name = T6rets_names[i]
    results[name] = [] # Initialise list
    for col in columns:
        t_statistic, p_value = stats.ttest_1samp(ret.dropna()[col], 0)
        results[name].append(t_statistic)

T6tstat = pd.DataFrame(results, index = columns).rename(columns={0: 't-statistic'}).T

T6stats = [sharpe_BOSESG,sharpe_BOSEnv,sharpe_BOSSoc,sharpe_BOSGov,
           sk_BOSESG, sk_BOSEnv, sk_BOSSoc, sk_BOSGov,
           maxdd_BOSESG, maxdd_BOSEnv, maxdd_BOSSoc, maxdd_BOSGov,
           T6tstat]

T6stats = (pd.concat(T6stats)).round(4)

T6stats.to_csv('T6stats.csv', sep = ',')

###############################################################################
###############################################################################

#### Table 7 Transaction costs ####

###############################################################################
###############################################################################

#standard
T7stddata = {}

columns = ['roe','mom','mktcap'] #factors

for col in columns:
    data = df['2006-01-01':].copy()
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
        4, labels = False, duplicates = 'drop')) # sort on factor
    data = data[np.isfinite(data['port'])]
    T7stddata[col] = (data)

#M1
T7M1data = {}

columns = ['roe','mom','mktcap'] #factors

for col in columns:
    data = df['2006-01-01':].copy()
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
        4, labels = False, duplicates = 'drop')) # sort on factor
    data = data[np.isfinite(data['port'])]
    T7M1data[col] = (data)

#M3
T7M3data = {}

columns = ['roe','mom','mktcap'] #factors

for col in columns:
    data = df['2006-01-01':].copy()
    data['cap'] = data.groupby(['id']).mktcap.shift()
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data = data[np.isfinite(data[col])]
    data['rank'] = data[col].rank(method='first')
    
    #sort portfolio on signal decile
    data['port'] = data.groupby(['date'])['rank'].transform(lambda x: pd.qcut(x,
         10, labels = False, duplicates = 'drop')) # sort on factors
    data = data.dropna(subset = ['port']) # drop if no score

    #sort portfolio on bm decile by port
    data['port1'] = data.groupby(['date', 'port'])[col].transform(lambda x: pd.qcut(x,
         10, labels = False, duplicates = 'drop')) # sort on bm 
    
    data['pport'] = (data['port']*10)
    del data['port']
    data['port1'] = data['pport'] + data['port1']
    data['port2'] = (0.5*data['ESG'] + 0.5*data['port1']) # interaction factor
    
    # sorting signal for interaction factor
    data['port'] = data.groupby(['date'])['port2'].transform(lambda x: pd.qcut(x,
         4, labels = False, duplicates = 'drop')) # sort on interaction factor
    data = (data[np.isfinite(data['port'])])
    T7M3data[col] = (data)


#create a dict with a key as the variable name
T7data = {'stdroe': T7stddata['roe'], 'stdmom':T7stddata['mom'],'stdmktcap':T7stddata['mktcap'],
          'M1roe':T7M1data['roe'],'M1mom':T7M1data['mom'],'M1mktcap':T7M1data['mktcap'],
          'M3roe':T7M3data['roe'],'M3mom':T7M3data['mom'],'M3mktcap':T7M3data['mktcap']}
  
def Turnover(df):
    data1 = pd.DataFrame(df.groupby(['date','port'])['port'].count())
    data1.columns.values[[0]] = ['turnover_monthly']
    df['turnover_monthly'] = df.groupby('id')['port'].diff().ne(0).astype(int).clip(upper = 1)
    data2 = pd.DataFrame(df.groupby(['date','port'])['turnover_monthly'].sum())
    turn = data2.div(data1,axis = 0)
    tmonth = turn.groupby(['port']).mean()
    tmonth.loc[tmonth.shape[0]] = ((tmonth.iloc[0]+tmonth.iloc[3])/2)
    tannual = (tmonth.copy()*12).rename(columns = {'turnover_monthly':'turnover_annual'})
    return tmonth, tannual

T7 = {k:Turnover(v) for k,v in T7data.items()}

'''This returns the portfolio turnovers for monthly and annually which are very 
similar to the result in paper (Only investigate successful strategies in paper, 
hence why min vol and value not included. Size is consistent, mom and quality 
similar. I am currently not sure how to get T7 into a dataframe so I can 
calculate the returns after accounting for transaction cost. But
since this turnover is very similar for mom and quality, I know the net returns 
will also be very similar'''

###############################################################################
###############################################################################

#### Figure 1 Scores distribution by sector ####

###############################################################################
###############################################################################
F1data = df.copy()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10,8), 
    sharex='col', sharey='row')
F1data.boxplot('ESG','sector', ax = ax1, vert= False)
ax1.set_title('')
ax1.set_xlabel('ESG')
ax1.xaxis.grid()
F1data.boxplot('Soc','sector', ax = ax2, vert= False)
ax2.set_title('')
ax2.set_xlabel('Soc')
ax2.xaxis.grid()
F1data.boxplot('Env','sector', ax = ax3, vert= False)
ax3.set_title('')
ax3.set_xlabel('Env')
ax3.xaxis.grid()
F1data.boxplot('Gov','sector', ax = ax4, vert= False)
ax4.set_title('')
ax4.set_xlabel('Gov')
ax4.xaxis.grid()
plt.suptitle("Figure 1 Scores distribution by sector", size=12)

###############################################################################
###############################################################################

#### Figure 3 Cumulative performance ####

###############################################################################
###############################################################################

asx200 = pd.read_csv('asx200.csv', index_col='date', parse_dates=True)
asx200.index = returns.index
F3data = returns.copy()
F3data= F3data.join(asx200)
F3logrets = np.log(1 + F3data)
logcumrets = np.exp(F3logrets.cumsum())

logcumrets.to_csv('logrets.csv', sep = ',')

roecumrets = logcumrets[['roeLS','M1roeLS','M2roeLS', 'M3roeLS','ASX200']]
momcumrets = logcumrets[['momLS','M1momLS','M2momLS', 'M3momLS','ASX200']]
sizecumrets = logcumrets[['mktcapLS','M1mktcapLS','M2mktcapLS', 'M3mktcapLS','ASX200']]

f, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize = (5,15))
roecumrets.plot(ax=ax1, title='Quality', style=[':', '--', '-','-.'])
ax1.set_xlabel('')
ax1.set_ylabel('Cumulative Return ($)')
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis
momcumrets.plot(ax=ax2, title='Momentum', style=[':', '--', '-','-.'])
ax2.set_xlabel('')
ax2.set_ylabel('Cumulative Return ($)')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis
sizecumrets.plot(ax=ax3, title='Size', style=[':', '--', '-','-.'])
ax3.set_ylabel('Cumulative Return ($)')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis

''' 
Plot looks a bit clunky with both simple/log rets - inconsistent cumulative 
return with results in Stata for size particularly. I believe this arises from 
the duplicates = 'drop' command needed when sorting portfolios. Need to include
this because a Python specific error arises if not included. 

Momentum similar, quality is slightly different too (higher cum ret in Python).
'''
 
###############################################################################
###############################################################################

#### Figure 4 Rolling performance ####

###############################################################################
###############################################################################

F4data = returns.copy()
F4data = F4data.join(asx200)

F4datarol = F4data.rolling(36).mean()
F4datarol = F4datarol['2009-01-01':]

roerolrets = F4datarol[['roeLS','M1roeLS','M2roeLS', 'M3roeLS','ASX200']]
momrolrets = F4datarol[['momLS','M1momLS','M2momLS', 'M3momLS','ASX200']]
sizerolrets = F4datarol[['mktcapLS','M1mktcapLS','M2mktcapLS', 'M3mktcapLS','ASX200']]

#plot
f, ((ax1, ax2,ax3)) = plt.subplots(3,1, figsize = (5,15))
roerolrets.plot(ax=ax1, title='Quality', style=[':', '--', '-','-.'])
ax1.set_xlabel('')
ax1.set_ylabel('Monthly Returns Over 3-years (%)')
ax1.axhline(0, color='black', lw=0.5)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis
momrolrets.plot(ax=ax2, title='Momentum', style=[':', '--', '-','-.'])
ax2.set_xlabel('')
ax2.set_ylabel('Monthly Returns Over 3-years (%)')
ax2.axhline(0, color='black', lw=0.5)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis
sizerolrets.plot(ax=ax3, title='Size', style=[':', '--', '-','-.'])
ax3.set_ylabel('Monthly Returns Over 3-years (%)')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Put a legend to the right of the current axis
ax3.axhline(0, color='black', lw=0.5)

'''graph comes up much better in Stata'''

###############################################################################
###############################################################################

#### Figure 5 Number of stocks traded ####

###############################################################################
###############################################################################

#Standard factors

long = {}
short = {}
countlong = {}
countshort = {}

columns = ['roe','mom','mktcap'] #factors

for col in columns:
    data = df.copy()
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    if col == 'mktcap':
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on mom factor
        long[col] = data.loc[data['port'] == 0.0]
        short[col] = data.loc[data['port'] == 3.0]
        countlong[col] = long[col].groupby(['date'])['port'].count()
        countshort[col] = short[col].groupby(['date'])['port'].count()
    else:
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop')) # sort on mom factor
        long[col] = data.loc[data['port'] == 3.0]
        short[col] = data.loc[data['port'] == 0.0]
        countlong[col] = long[col].groupby(['date'])['port'].count()
        countshort[col] = short[col].groupby(['date'])['port'].count()

#M1
        
M1long = {}
M1short = {}
M1countlong = {}
M1countshort = {}

for col in columns:
    data = df.copy()
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    if col == 'mktcap':
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop'))
        M1long[col] = data.loc[data['port'] == 0.0]
        M1short[col] = data.loc[data['port'] == 3.0]
        M1countlong[col] = (M1long[col].groupby(['date'])['port'].count()).rename({'port':'M1'+col})
        M1countshort[col] = (M1short[col].groupby(['date'])['port'].count()).rename({'port':'M1'+col})
    else:
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop'))
        M1long[col] = data.loc[data['port'] == 3.0]
        M1short[col] = data.loc[data['port'] == 0.0]
        M1countlong[col] = (M1long[col].groupby(['date'])['port'].count()).rename({'port':'M1'+col})
        M1countshort[col] = (M1short[col].groupby(['date'])['port'].count()).rename({'port':'M1'+col})

#M2
        
M2long = {}
M2short = {}
M2countlong = {}
M2countshort = {}

for col in columns:
    data = df.copy()
    data['cap'] = data.groupby(level = ['id']).mktcap.shift()
    if col == 'mktcap':
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop'))
        data = data.dropna(subset = ['ESG']) # drop if no ESG score
        M2long[col] = data.loc[data['port'] == 0.0]
        M2short[col] = data.loc[data['port'] == 3.0]
        M2countlong[col] = M2long[col].groupby(['date'])['port'].count()
        M2countshort[col] = M2short[col].groupby(['date'])['port'].count()
    else:
        data['port'] = data.groupby(['date'])[col].transform(lambda x: pd.qcut(x, 
            4, labels = False, duplicates = 'drop'))  
        data = data.dropna(subset = ['ESG']) # drop if no ESG score
        M2long[col] = data.loc[data['port'] == 3.0]
        M2short[col] = data.loc[data['port'] == 0.0]
        M2countlong[col] = M2long[col].groupby(['date'])['port'].count()
        M2countshort[col] = M2short[col].groupby(['date'])['port'].count()

#M3
        
M3long = {}
M3short = {}
M3countlong = {}
M3countshort = {}

for col in columns:
    data = df.copy()
    data['cap'] = data.groupby(['id']).mktcap.shift()
    data = data.dropna(subset = ['ESG']) # drop if no ESG score
    data = data[np.isfinite(data[col])]
    data['rank'] = data[col].rank(method='first')
    
    #sort portfolio on signal decile
    data['port'] = data.groupby(['date'])['rank'].transform(lambda x: pd.qcut(x,
         10, labels = False, duplicates = 'drop')) # sort on factors
    data = data.dropna(subset = ['port']) # drop if no score

    #sort portfolio on bm decile by port
    data['port1'] = data.groupby(['date', 'port'])[col].transform(lambda x: pd.qcut(x,
         10, labels = False, duplicates = 'drop')) # sort on bm 
    
    data['port'] = data['port']*10
    data['port1'] = data['port'] + data['port1']
    data['port2'] = (0.5*data['ESG'] + 0.5*data['port1']) # interaction factor
    
    # sorting signal for interaction factor
    data['sort'] = data.groupby(['date'])['port2'].transform(lambda x: pd.qcut(x,
         4, labels = False, duplicates = 'drop')) # sort on interaction factor
    if col == 'mktcap':
        M3long[col] = data.loc[data['sort'] == 0.0]
        M3short[col] = data.loc[data['sort'] == 3.0]
        M3countlong[col] = M3long[col].groupby(['date'])['sort'].count()
        M3countshort[col] = (M3short[col].groupby(['date'])['sort'].count())
    else:
        M3long[col] = data.loc[data['sort'] == 3.0]
        M3short[col] = data.loc[data['sort'] == 0.0]
        M3countlong[col] = M3long[col].groupby(['date'])['sort'].count()
        M3countshort[col] = M3short[col].groupby(['date'])['sort'].count()

#long
roelong = pd.concat([countlong['roe'],M1countlong['roe'],
                     M2countlong['roe'],M3countlong['roe']], axis=1)
roelong.columns = ['std', 'M1', 'M2', 'M3']

momlong = pd.concat([countlong['mom'],M1countlong['mom'],
                     M2countlong['mom'],M3countlong['mom']], axis=1)
momlong.columns = ['std', 'M1', 'M2', 'M3']

mktcaplong = pd.concat([countlong['mktcap'],M1countlong['mktcap'],
                        M2countlong['mktcap'],M3countlong['mktcap']], axis=1)
mktcaplong.columns = ['std', 'M1', 'M2', 'M3']

#short
roeshort = pd.concat([countshort['roe'],M1countshort['roe'],
                     M2countshort['roe'],M3countshort['roe']], axis=1)
roeshort.columns = ['std', 'M1', 'M2', 'M3']

momshort = pd.concat([countshort['mom'],M1countshort['mom'],
                     M2countshort['mom'],M3countshort['mom']], axis=1)
momshort.columns = ['std', 'M1', 'M2', 'M3']

mktcapshort = pd.concat([countshort['mktcap'],M1countshort['mktcap'],
                        M2countshort['mktcap'],M3countshort['mktcap']], axis=1)
mktcapshort.columns = ['std', 'M1', 'M2', 'M3']


#plot stock count portfolio 
f, ((ax1, ax2) , (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize = (8,15),sharey='row')
roelong.plot(ax=ax1, title='Qualty Long', style=[':', '--', '-','-.'])
ax1.set_xlabel('')
ax1.set_ylabel('No. of stocks in Portfolio')
roeshort.plot(ax=ax2, title='Qualty Short', style=[':', '--', '-','-.'])
ax2.set_xlabel('')
momlong.plot(ax=ax3, title='Momentum Long', style=[':', '--', '-','-.'])
ax3.set_xlabel('')
ax3.set_ylabel('No. of stocks in Portfolio')
momshort.plot(ax=ax4, title='Momentum Short', style=[':', '--', '-','-.'])
ax4.set_xlabel('')
mktcaplong.plot(ax=ax5, title='Size Long', style=[':', '--', '-','-.'])
ax5.set_ylabel('No. of stocks in Portfolio')
ax5.set_xlabel('')
mktcapshort.plot(ax=ax6, title='Size Short', style=[':', '--', '-','-.'])
ax6.set_xlabel('')

'''graph comes up much better in Stata - not so clunky'''

###############################################################################
###############################################################################

#### Figure 6 Holdings by sector ####

###############################################################################
###############################################################################

#compare standard factors holdings with interaction factor holdings (M3)

#create a dict 
dfs = {'stdroeL':long['roe'], 'stdroeS':short['roe'],
       'M3roeL':M3long['roe'], 'M3roeS':M3short['roe'],
       'stdmomL':long['mom'], 'stdmomS':short['mom'],
       'M3momL':M3long['mom'], 'M3momS':M3short['mom'],
       'stdmktcapL':long['mktcap'], 'stdmktcapS':short['mktcap'],
       'M3mktcapL':M3long['mktcap'], 'M3mktcapS':M3short['mktcap']}

#function to determine the holdings by sector         
def Holdings(df):
    df = df['2006-01-01':].reset_index()
    df['count'] = 1
    df = pd.DataFrame(df.groupby(['sector','date']).count())
    df['percentage'] = df['count'].div(df.groupby('date')['count'].transform('sum')).mul(100)
    return df['percentage'].unstack()

#dict comprehension
F6 = {k:Holdings(v) for k,v in dfs.items()}

#access the desired df, for example standard roe 
stdlongroe = F6['stdroeL'].T

#make the stacked plot 
'''Had a go at this in Python but the output is similar to what it is in Stata,
not very good (In this case looks pretty stupid). For the paper, I imported the 
sector holdings into excel and made a stacked plot, which looks much better. '''

'''plt.stackplot(stdlongroe['Consumer Discretionary'],stdlongroe['Consumer Staple'],  
              stdlongroe['Energy'], stdlongroe['Financials'], stdlongroe['Health Care'],
              stdlongroe['Industrials'],stdlongroe['Information Technology'],
              stdlongroe['Materials'], stdlongroe['Telecommunications'],
              stdlongroe['Utilities'],
              labels=['Consumer Discretionary',
                        'Consumer Staple','Energy','Financials','Health Care',
                        'Industrials', 'Information Technology', 'Materials',
                        'Telecommunications','Utilities'])
plt.margins(0,0)
plt.show()'''

###############################################################################
###############################################################################