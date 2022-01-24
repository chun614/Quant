" The following program is used to show the econometric model construction towards the census data processed in 'Data_Process.py'. See 'WorkingPaper_Conc.pdf', which is available on the application platform, for details of the model employed." 
# Copyright (C) 2021 Chun Hung, Tsang


### Enviornment
import numpy as np
import pandas as pd
import os
import re
from copy import deepcopy
import math
from sklearn import preprocessing
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import missingno as msno
%matplotlib inline

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)

## Set the path of data
_Path = r'D:\Program Files (x86)\Project Conc'
os.chdir(_Path)



### Statistical Preprocess
## Add a column for 2-digit sectors.
SameInd_90 = SameInd_90.reset_index()
SameInd_95 = SameInd_95.reset_index()
SameInd_100 = SameInd_100.reset_index()

SameInd_90['2-digit Sector'] = SameInd_90['Code']
SameInd_90['2-digit Sector'] = SameInd_90['2-digit Sector'].apply(lambda x: int(str(x)[:-2]))

SameInd_95['2-digit Sector'] = SameInd_95['Code']
SameInd_95['2-digit Sector'] = SameInd_95['2-digit Sector'].apply(lambda x: int(str(x)[:-2]))

SameInd_100['2-digit Sector'] = SameInd_100['Code']
SameInd_100['2-digit Sector'] = SameInd_100['2-digit Sector'].apply(lambda x: int(str(x)[:-2]))


## Fit the top level sector names to data based on the industrial codes.
def Func_FitSector(DGBAS, Sector):
    ## Create a column for putting sector names which are going to be integrated into main data.
    DGBAS['Top-lv Sector'] = ''
    
    Code = DGBAS['Code']
    
    for rowSector in range(len(Sector)):
        DGBAS.loc[:,'Top-lv Sector'][(Code >= Sector['Start Index'][rowSector]) == (Code <= Sector['End Index'][rowSector])] = Sector.iloc[rowSector, 3]

Func_FitSector(SameInd_90, df_PI)
Func_FitSector(SameInd_95, df_PI)
Func_FitSector(SameInd_100, df_PI)


## Reset the industrial codes back to index.
SameInd_90 = SameInd_90.set_index('Code')
SameInd_95 = SameInd_95.set_index('Code')
SameInd_100 = SameInd_100.set_index('Code')


## To build 5-year difference matrics for log and linear regressions among census data respectively.
def Func_5yrDiff(lstOfYrData, cat, dummy, model = 'log'):    
    ## Split the variables which would not join in the process of taking log difference:
    #    Categorical variables                  : 2-digit Sector, Top-lv Sector;
    #    Dummy variable (since existing 0)      : R&D, IT, Info (IT);
    cols = lstOfYrData[1].columns.tolist()
    for var in cat:
        cols.remove(var)
    for var in dummy:
        if var in cols:
            cols.remove(var)
    
    Census = []
    for yr in range( len(lstOfYrData) ):
        Census.append( lstOfYrData[yr][cols] )

    ## Decide whether to take log on data matrics.
    if model == 'log':
        Census = [ np.log(yrData) for yrData in Census ]
    elif model == 'linear':
        pass

    ## Calculate the 5-yr difference.
    DiffCensus = []
    for yr in range( len(Census) - 1 ):
        DiffCensus.append( Census[yr+1] - Census[yr] )
        
        
    ## Add back the categorical variables splited before for each 5-yr difference matrics.
    for yr in range( len(DiffCensus) ):
        for var in cat:
            DiffCensus[yr][var] = lstOfYrData[ yr + 1 ][var]    
    
    return DiffCensus


## Create different dataframes for different groups of sectors.
def Func_SelectSector(lstOfYrData, sectorName = 'All'):
    lstOfSectorData = []
    
    # A. All sectors.
    # Manufacturing, Wholesale, Retail, Professional, Scientific, & Technical Services,
    # Transportation & Warehousing, Publication & Info, Art & Leisure, Asset Transaction and Leasing,
    # Health Care, Mining, Accommodation & Catering, Utilities, Construction, Other Services.
    if sectorName == 'All':
        lstOfSectorData = lstOfYrData
        
    # B. Manufacturing sector.
    elif sectorName == 'Manufacturing':
        for yrData in lstOfYrData:
            lstOfSectorData.append( yrData[ yrData['Top-lv Sector'] == 'Manufacturing' ] )
            
    # C. Non-manufacturing sector.
    elif sectorName == 'NonManufacturing':
        for yrData in lstOfYrData:
            lstOfSectorData.append( yrData[ yrData['Top-lv Sector'] != 'Manufacturing' ] )

    # D. Services sector (whole tertiary sector).
    elif sectorName == 'Services':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] != 'Mining' ]
            sector = sector[ sector['Top-lv Sector'] != 'Manufacturing' ]
            sector = sector[ sector['Top-lv Sector'] != 'Utilities' ]
            sector = sector[ sector['Top-lv Sector'] != 'Construction' ]
            lstOfSectorData.append( sector )

    # E. Autor's 6 sectors.
    elif sectorName == 'Autor':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] != 'Mining' ]
            sector = sector[ sector['Top-lv Sector'] != 'Construction' ]
            lstOfSectorData.append( sector )

    # F. Autor's 6 sector category - Finance.
    elif sectorName == 'Finance':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Asset Transaction and Leasing' ]
            lstOfSectorData.append( sector )

    # G. Autor's 6 sector category - Wholesale.
    elif sectorName == 'Wholesale':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Wholesale' ]
            lstOfSectorData.append( sector )

    # H. Autor's 6 sector category - Retail.
    elif sectorName == 'Retail':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Retail' ]
            lstOfSectorData.append( sector )

    # I. Autor's 6 sector category - UtilitiesAndTransportation.
    elif sectorName == 'UtilitiesAndTransportation':
        ## List out the names of constituent sub-sectors in current aggregated sector category.
        constituents = ['Utilities', 'Transportation & Warehousing']
        for yrData in lstOfYrData:
            ## List out the sub-sectors that are not belong to current aggregated sector category.
            removedSubsectors = [ subsector for subsector in yrData['Top-lv Sector'].unique() if subsector not in constituents ]
            # The data to be removing sub-sectors assigned for current period.
            sector = yrData
            for subsector in removedSubsectors:
                ## Remove the data whose sub-sectors that are not belong to current aggregated sector category.
                sector = sector[ sector['Top-lv Sector'] != subsector ]
            ## Combine the current period data with data of all other periods.
            lstOfSectorData.append( sector )

    # J. Autor's 6 sector category - Service, or other definition, according to its constituents set.
    # See the definition adopted here by searching: ----- Define service sector -----
    elif sectorName == 'Service':
        ## List out the names of constituent sub-sectors in current aggregated sector category.
        constituents = _serviceSector
        for yrData in lstOfYrData:
            ## List out the sub-sectors that are not belong to current aggregated sector category.
            removedSubsectors = [ subsector for subsector in yrData['Top-lv Sector'].unique() if subsector not in constituents ]
            # The data to be removing sub-sectors assigned for current period.
            sector = yrData
            for subsector in removedSubsectors:
                ## Remove the data whose sub-sectors that are not belong to current aggregated sector category.
                sector = sector[ sector['Top-lv Sector'] != subsector ]
            ## Combine the current period data with data of all other periods.
            lstOfSectorData.append( sector )

    # K. 7th sector out of Autor's 6 sector category - Construction.
    elif sectorName == 'Construction':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Construction' ]
            lstOfSectorData.append( sector )

    # L. Abraham & Bormans (2020)'s sector category - Transportation & Storage.
    elif sectorName == 'Transportation & Warehousing':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Transportation & Warehousing' ]
            lstOfSectorData.append( sector )

    # M. Abraham & Bormans (2020)'s sector category - Info & Communication.
    elif sectorName == 'Publication & Info':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Publication & Info' ]
            lstOfSectorData.append( sector )

    # N. Abraham & Bormans (2020)'s sector category - Professional, Scientific, & Technical Services.
    elif sectorName == 'Professional, Scientific, & Technical Services':
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == 'Professional, Scientific, & Technical Services' ]
            lstOfSectorData.append( sector )

    # O. Abraham & Bormans (2020)'s sector category - Wholesale & Retail.
    elif sectorName == 'WholesaleAndRetail':
        ## List out the names of constituent sub-sectors in current aggregated sector category.
        constituents = ['Wholesale', 'Retail']
        for yrData in lstOfYrData:
            ## List out the sub-sectors that are not belong to current aggregated sector category.
            removedSubsectors = [ subsector for subsector in yrData['Top-lv Sector'].unique() if subsector not in constituents ]
            # The data to be removing sub-sectors assigned for current period.
            sector = yrData
            for subsector in removedSubsectors:
                ## Remove the data whose sub-sectors that are not belong to current aggregated sector category.
                sector = sector[ sector['Top-lv Sector'] != subsector ]
            ## Combine the current period data with data of all other periods.
            lstOfSectorData.append( sector )

    # P. For aggregated sectors.
    elif type(sectorName) == list:
        for yrData in lstOfYrData:
            ## List out the sub-sectors that are not belong to current aggregated sector category.
            removedSubsectors = [ subsector for subsector in yrData['Top-lv Sector'].unique() if subsector not in sectorName ]
            # The data to be removing sub-sectors assigned for current period.
            sector = yrData
            for subsector in removedSubsectors:
                ## Remove the data whose sub-sectors that are not belong to current aggregated sector category.
                sector = sector[ sector['Top-lv Sector'] != subsector ]
            ## Combine the current period data with data of all other periods.
            lstOfSectorData.append( sector )
    
    # Q. For remaining individual sectors specified.
    else:
        for yrData in lstOfYrData:
            sector = yrData[ yrData['Top-lv Sector'] == sectorName ]
            lstOfSectorData.append( sector )

    return lstOfSectorData


## Function for create dummies according to catagorical data existed.
def Func_Dummy(Census, Control):
    for i in Control.unique().tolist():
        Census[i] = 0
        Census.loc[:, i][ Control == i ] = 1


def Func_Main_AddDummy(lstOfYrData):
    for yr in range( len(lstOfYrData) ):
        ## Add dummies for top sector catagory for the sake of separately analysis w.r.t. top sectors.
        Func_Dummy(lstOfYrData[yr], lstOfYrData[yr]['Top-lv Sector'])

        ## Create dummies for 2-digit sectors for the sake of control fixed effect.
        Func_Dummy(lstOfYrData[yr], lstOfYrData[yr]['2-digit Sector'])

        ## Drop one dummy of 2-digit sectors in order to avoid collinearity.
        lstOfYrData[yr] = lstOfYrData[yr].iloc[:, :-1]
        
        ## Turn the index as entity dummies.
        lstOfYrData[yr] = lstOfYrData[yr].reset_index()
    
    
    ## Add time stamp.
    # Drop 1 year dummy, Year_95to100, to avoid collinearity with constant term.
    lstOfYrData[0]['Year_90to95'] = 1
    lstOfYrData[1]['Year_90to95'] = 0



### Model Building
def Func_Main_InputBuilding(lstOfYrData, model = 'log', sector = 'All'):
    ## Execute to build 5-year difference matrics.
    # cat represents the categorical variables ('2-digit Sector', 'Top-lv Sector') and dummy represents the variables could not be applied log-difference directly.
    lst_Diff_Ind = Func_5yrDiff(lstOfYrData, ['2-digit Sector', 'Top-lv Sector'], 
                                ['R.Profit', 'Avg Profit (per L)', 'Avg Profit (per K)', 'Profit rate', 
                                 'N.Profit', 'N.Avg Profit (per L)',
                                 'R&D', 'IT', 'Info (IT)'], model)
    
    ## Execute to create dataframe for the set of sectors selected.
    lst_Diff_Ind = Func_SelectSector(lst_Diff_Ind, sector)
    
    ## Execute to create dummies according to catagorical data.
    Func_Main_AddDummy(lst_Diff_Ind)

    ## Combine the panel data tgt across all yrs (after sector selection in previous section).
    Panel = pd.DataFrame()
    for yrData in lst_Diff_Ind:
        Panel = Panel.append( yrData )
    
    return Panel.reset_index(drop=True)


## Put all census data across years into a list.
_Census = deepcopy([SameInd_90, SameInd_95, SameInd_100])

## Set initial parameters.
_regressModelSpec = 'log'
_covType = 'HC0'                                     # Set the mode of standard errors
_std = True                                          # Standardization
_serviceSector = ['Accommodation & Catering', 'Health Care', 'Art & Leisure', 'Other Services']

_sectorsReg = ['Manufacturing', 'Wholesale', 'Retail', 'Transportation & Warehousing', 'Construction',
               'Publication & Info', 'Professional, Scientific, & Technical Services', 'Finance',
               ['Accommodation & Catering', 'Health Care', 'Art & Leisure', 'Other Services'], 'Utilities'
              ]

_Matrix_All                = Func_Main_InputBuilding(_Census, _regressModelSpec, 'All')
_Matrix_Manufacturing      = Func_Main_InputBuilding(_Census, _regressModelSpec, 'Manufacturing')
_Matrix_NonManufacturing   = Func_Main_InputBuilding(_Census, _regressModelSpec, 'NonManufacturing')
_Matrix_Services           = Func_Main_InputBuilding(_Census, _regressModelSpec, 'Services')


## Set Regressors.
TrueRegressors = [['CR4', 'Avg Inc (per L)'],
                  ['CR4', 'Avg Inc (per L)', 'EntryPC'],
                  ['CR4', 'CR4^2', 'Avg Inc (per L)', 'Avg Inc (per L)^2', 'EntryPC'],
                 ]

## Since the high power terms are not generated yet, these have to be removed first to compile the regression space.
_ReducedRegressors = deepcopy(TrueRegressors)
# -$$$ NOTE $$$- 2 regression spaces are prepared as high power terms have to be created after standardization.
# High power terms would be degenerated to linear terms through standardiztion whatever they are created either industrialized data or taking 5-yr difference so they can only be created after standardization.
# However, the regressors have to be chosen from dataset, including high power terms of variable, which are not existed in data yet, into standardization and regression. Hence, the regression space has to remove the high power terms first to extract the data of regressors into standardization from dataset first and then create the high power terms in the standardized data so that regression models are correctly specified.
# TrueRegressors and TrueRegSpace correspond to complete sets of regressors with high power terms input, whereas _ReducedRegressors and _ReducedRegSpace correspond to reduced sets of regressors without high power terms.

## Set Regressands.
# -$$$ NOTE $$$- In log-regression setting, profit would be removed out from regressands as it has negative values before taking log.
_Regressands = ['R.Inc',
                'PI',
                'CR4',
                'Avg Inc (per L)',
                'R.Entry.Inc',
                'N.Inc',
                'Avg Wage (per L)',
                'Labor',
                'R.Wage',
                'Wage share',
                ]


## Extract the high power terms of variable in regressors which are not generated yet.
_powerTerms = []
## For every set of X,
for _X in range(len( _ReducedRegressors )):
    ## For each variable in current set of regressors,
    for x_k in range(len( _ReducedRegressors[_X] )):
        ## When this regressor is high power term of variable,
        if '^' in _ReducedRegressors[_X][x_k]:
            ## Record the name and position of the regressor.
            _powerTerms.append( (_ReducedRegressors[_X][x_k], _X, x_k) )

## Temporarily remove the high power terms since they are not yet existed in X and hence it could not be standardized in next procedure.
# This step (remove) cannot combine with step of searching terms above since removing items in X will affect the length of regressors with inconsistence of the for-loop times set and wrongly change the position info of high power term(s) afterward.
for _term in _powerTerms:
    _var, _p1, _p2 = _term
    _ReducedRegressors[_p1].remove( _var )


## Create matrix space for variables of interest for each combination of regressand and set of regressors.
def Func_RegSpace(regressors):
    RegressionSpace = []
    ## For every set of X,
    for _X in regressors:
        ## Combine each of regressand with current set of X.
        _RegressionSpace_df = pd.DataFrame({'Y' : _Regressands})
        for x_k in _X:
            _RegressionSpace_df[ x_k ] = x_k
    
        ## Convert the sub-matrix space to list.
        _RegressionSpace_n = []
        for i in range( len(_RegressionSpace_df) ):
            _RegressionSpace_n.append( _RegressionSpace_df.iloc[i].tolist() )
    
        ## Inlay the sub-matrix space into the matrix space.
        RegressionSpace.append( _RegressionSpace_n )

    return RegressionSpace

_ReducedRegSpace = Func_RegSpace(_ReducedRegressors)
TrueRegSpace     = Func_RegSpace(TrueRegressors)


## Once a regressor also becomes the regressand, it is removed from the regressors.
def Func_CheckRepeatedVariable(lstOfVarOfInterest, lstOfRegressors):
    for x in lstOfRegressors:
        if lstOfVarOfInterest.count(x) > 1:
            lstOfVarOfInterest.reverse()
            lstOfVarOfInterest.remove(x)     # Since remove() deletes the first element encountered.
            lstOfVarOfInterest.reverse()

for n in range( len(TrueRegSpace) ):
    for lstOfVarOfInterest in TrueRegSpace[n]:
        Func_CheckRepeatedVariable( lstOfVarOfInterest, TrueRegressors[n] )

for n in range( len(_ReducedRegSpace) ):
    for lstOfVarOfInterest in _ReducedRegSpace[n]:
        Func_CheckRepeatedVariable( lstOfVarOfInterest, _ReducedRegressors[n] )


## Standardize the variables of each census year data or all census years together.
def Func_STD(data, Var):
    ## Select regressors.
    X = data[Var]
    
    ## Standardize the variables.
    _columns = X.columns
    _index = X.index
    stdn = preprocessing.StandardScaler()
    # Determine whether apply standardization.
    if _std:
        X = stdn.fit_transform(X)
    X = pd.DataFrame(X, index = _index, columns = _columns)
    
    return X


def Func_Main_ExecSTD(data):
    ## Execute standardization.
    X = []
    ## All regressands are standardized with every set of regressors in turn.
    for subMatrixSpace in _ReducedRegSpace:
        X_n = []
        ## Every regressand is standardized with current set of regressors.
        for lstOfVarOfInterest in subMatrixSpace:
            x_k = Func_STD(data, lstOfVarOfInterest)
            ## Inlay the standardized data of each regression into sub-matrix spaces.
            X_n.append( x_k )
        
        ## Inlay the standardized sub-matrix spaces into the matrix space.
        X.append(X_n)

    return X

X_All              = Func_Main_ExecSTD(_Matrix_All)
X_Manufacturing    = Func_Main_ExecSTD(_Matrix_Manufacturing)
X_NonManufacturing = Func_Main_ExecSTD(_Matrix_NonManufacturing)
X_Services         = Func_Main_ExecSTD(_Matrix_Services)


## Generate high power terms of variable according to info recorded in previous step.
def Func_PowerTerm(data):
    for _term in _powerTerms:
        _var, _p1, _p2 = _term
        # The first element is the linear term of the regressor and the second element is the power term.
        _x_k = _var.split('^')
        
        for y in range(len( data[_p1] )):
            ## Create high power term column.
            data[_p1][y][ _var ] = data[_p1][y][ _x_k[0] ] ** int(_x_k[1])
    
    ## After all high power terms being generated, reorder the order of variables of interest according to original (true) regression space.
    for subMatrixSpace in range(len( TrueRegSpace )):
        for lstOfVarOfInterest in range(len( TrueRegSpace[subMatrixSpace] )):
            data[subMatrixSpace][lstOfVarOfInterest] = \
                    data[subMatrixSpace][lstOfVarOfInterest][  TrueRegSpace[subMatrixSpace][lstOfVarOfInterest]  ]

    return data

X_All              = Func_PowerTerm(X_All)
X_Manufacturing    = Func_PowerTerm(X_Manufacturing)
X_NonManufacturing = Func_PowerTerm(X_NonManufacturing)
X_Services         = Func_PowerTerm(X_Services)


### Model Running
# The following code is based on the standardization of all observation across all years together.
def Func_OLSInput(data, Var, mode = 'all', ts = False):
    # For the case of regressing the data of a specific sector, the sector dummies would not need to add.
    if mode != 'sector':
        ## Add the column of 2-digit sector to control fixed effect (already drop 1 dummy in previous function).
        # Width of column of time dummy is less than length of census years by 2 as less of 1 for taking difference and less of 1 for dropping 1 time dummy.
        X = Var.join( data.iloc[ :,  -len(data['2-digit Sector'].unique()) : -( len(_Census) - 2 ) ] )
    else:
        X = Var
    
    # -$$$ NOTE $$$- For panel data only.
    ## Add the column of time dummy to control time effect.
    X = X.join( data.iloc[:, -( len(_Census) - 2 )] )
    
    ## In case of the 'time series' parameter to be True, it regresses the Y against X with lagging 1 period.
    if ts == True:
        FullMatrix = X
        # Regressors subject to period of difference between year 95 and 90.
        X = FullMatrix[ FullMatrix['Year_90to95'] == 1 ]
        # Regressands subject to period of difference between year 100 and 95.
        Y = FullMatrix[ FullMatrix['Year_90to95'] == 0 ]
        ## Divide regressors and regressand.
        Y = Y.iloc[:, 0].reset_index(drop=True)
        X = X.iloc[:, 1:-1].reset_index(drop=True)  # The redundant time dummy is removed.
    # For the case no lag period is controlled.
    else: 
        ## Divide regressors and regressand.
        Y = X.iloc[:, 0]
        X = X.iloc[:, 1:]

    ## Add constant term.
    X['const'] = 1
        
    ## Combine regressors and regressands respectively to form a complete panel interests.
    return Y, X


def Func_Main_OLS(data, X, mode = 'all', ts = False):
    ## The regression results would be stored in a list.
    OLSresults = []
    
    for subMatrixSpace in X:
        OLSresults_n = []
        
        for lstOfVarOfInterest in subMatrixSpace:
            y, x = Func_OLSInput(data, lstOfVarOfInterest, mode = mode, ts = ts)
            result = sm.OLS( y, x ).fit( cov_type = _covType )
            OLSresults_n.append( result )

        OLSresults.append( OLSresults_n )

    return OLSresults

# -$$$ NOTE $$$- For single sector regression, the mode of Func_OLSInput() should be 'sector' s.t. the sector dummy would not be added into dataset, in order to avoid the issue of perfect collinearity with the constant term.
_results_All              = Func_Main_OLS(_Matrix_All,              X_All)
_results_Manufacturing    = Func_Main_OLS(_Matrix_Manufacturing,    X_Manufacturing, mode = 'sector')
_results_NonManufacturing = Func_Main_OLS(_Matrix_NonManufacturing, X_NonManufacturing)
_results_Services         = Func_Main_OLS(_Matrix_Services,         X_Services)
# Lagging with 1 period:
_results_All_ts           = Func_Main_OLS(_Matrix_All,              X_All, ts = True)

# The 1st [parenthesis] represents set of regressors and the 2nd [parenthesis] represents regressand.
#print(_results_All[1][0].summary())
#_results_All[1][0].get_robustcov_results(cov_type='HC3').summary()
#_results_All_ts[4][2].summary()



### Data Postprocess
## Set the names of column (and width of table).
# Use names adopted by Ganapati (2020) and Olley & Pakes (1996).
_GanapatiOPName = ['Output', 'Price', 'Revenue', 'R.Labor Productivity (G)', 
                   'Mean Wage', 'Employees', 'Payroll', 'Labor Share',
                   'N.Labor Productivity (G)', '4-Firm Share',
                   'R.Labor Productivity (OP)', 'R.Institutional Productivity (OP)', 'R.Allocative Productivity (OP)',
                   'N.Labor Productivity (OP)', 'N.Institutional Productivity (OP)', 'N.Allocative Productivity (OP)']

_VarList = ['R.Inc', 'PI', 'N.Inc', 'Avg Inc (per L)',
            'Avg Wage (per L)', 'Labor', 'R.Wage', 'Wage share',
            'N.Avg Inc (per L)', 'CR4',
            'R.WAvg L Productivity', 'R.UWM L Productivity', 'R.Cov Prodty&Inc',
            'N.WAvg L Productivity', 'N.UWM L Productivity', 'N.Cov Prodty&Inc']

## Build up the table of conversion for item names.
_NameList = pd.DataFrame({'MyName'   : _VarList,
                    'GanapatiOPName' : _GanapatiOPName })


## Execute the conversion of item names according to either the table of conversion or a general rule.
def Func_ItemName(regvar, axis):
    lst = []
    for var in regvar:
        ## Apply conversion based on the table.
        for name in range( len(_NameList) ):
            if var == _NameList.iloc[name, 0]:
                var = _NameList.iloc[name, 1]
        
        ## Apply the general rule.
        if 'Ln' in var:
            lst.append( '△ ' + var )
        else:
            if _regressModelSpec == 'log':
                lst.append( '△ Ln ' + var )
            else:
                lst.append( '△ ' + var )
    
    ## Add 'Std' for item names in column.
    if axis == 'x':
        lst = [ 'Std ' + name for name in lst ]
        
    ## Use power bracket for high power terms.
    lst = [ '(' + name[:-2] + ')' + name[-2:] if '^' in name else name for name in lst ]
            
    return lst

## Generate column names according to regressands adopted in regression.
_ResultTable_Col = pd.DataFrame(columns = ['Result Table'] + Func_ItemName(_Regressands, 'y'))


## Every set of regressors and regressands would form a individual regression result table and all of the tables generated are put into a list under a set of sectors.
_ResultTable_empty = []
for _X_n in TrueRegressors:
    ## Make ready to put the row of current regressors with the column names set above into result table.
    _ResultTable_n = _ResultTable_Col.copy()

    ## Set the names of row.
    _IndexName = Func_ItemName(_X_n, 'x')
    
    ## Put the names of row to the result table accordingly.
    _RowName = [_IndexName[0], '']
    for i in range( 1, len(_X_n) ):
        _RowName.extend( [_IndexName[i], ''] )

    #_RowName.extend(['r2', 'Observations'])
    _RowName.extend(['const', '', 'r2', 'Observations'])

    _ResultTable_n.loc[:, 'Result Table'] = pd.Series( _RowName )
    
    ## Inlay the table with current regressor names into the list of result tables.
    _ResultTable_empty.append( _ResultTable_n )


## All sets of sectors share same table column and row names (i.e. Same variables of interest in regression).
ResultTable_All              = deepcopy(_ResultTable_empty)
ResultTable_Manufacturing    = deepcopy(_ResultTable_empty)
ResultTable_NonManufacturing = deepcopy(_ResultTable_empty)
ResultTable_Services         = deepcopy(_ResultTable_empty)
# Lagging with 1 period:
ResultTable_All_ts           = deepcopy(_ResultTable_empty)

## Add '*' based on significance.
def Func_FitStar(num, col, result, Table):
    if abs(list(result.tvalues)[num]) >= 2.575:
            Table.loc[num*2, col] = Table.loc[num*2, col] + '***'
    elif abs(list(result.tvalues)[num]) >= 1.96:
            Table.loc[num*2, col] = Table.loc[num*2, col] + '**'
    elif abs(list(result.tvalues)[num]) >= 1.645:
            Table.loc[num*2, col] = Table.loc[num*2, col] + '*'
    else:
        pass

## Fit the statistics of OLS results into table one by one.
def Func_FitTableCol(col, result, Table, X_order):
    # Marginal effects.
    for i in range( len( TrueRegressors[X_order] ) ):
        Table.loc[i*2, col]     = format(list(result.params)[i], "<5.4f")
        Table.loc[i*2 + 1, col] = '(' + format(list(result.bse)[i], "<5.4f") + ')'
    
    # Constant term.
    Table.loc[ len( TrueRegressors[X_order] ) * 2, col ]     = format(list(result.params)[-1], "<5.4f")
    Table.loc[ len( TrueRegressors[X_order] ) * 2 + 1, col ] = '(' + format(list(result.bse)[-1], "<5.4f") + ')'
    # R squared.
    Table.loc[ len( TrueRegressors[X_order] ) * 2 + 2, col ] = format(result.rsquared, "<5.4f")
    # Num of observations.
    Table.loc[ len( TrueRegressors[X_order] ) * 2 + 3, col ] = format(result.nobs, "<1.0f")
    
    #Table.loc[ len( TrueRegressors[X_order] ) * 2, col ]     = format(result.rsquared, "<5.4f")     # R squared.
    #Table.loc[ len( TrueRegressors[X_order] ) * 2 + 1, col ] = format(result.nobs, "<1.0f")         # Num of observations.
    
    for i in range( len( TrueRegressors[X_order] ) ):
        Func_FitStar(i, col, result, Table)


def Func_Main_FitTable(results, Table):
    ## Fit the results of current set of regressors to table.
    for n in range( len(Table) ):
        ## Fit the result of each regressand with current set of regressors to table.
        for i in range( len( Table[n].columns ) - 1 ):
            Func_FitTableCol( Table[n].columns[i+1], results[n][i], Table[n], n )

Func_Main_FitTable(_results_All, ResultTable_All)
Func_Main_FitTable(_results_Manufacturing, ResultTable_Manufacturing)
Func_Main_FitTable(_results_NonManufacturing, ResultTable_NonManufacturing)
Func_Main_FitTable(_results_Services, ResultTable_Services)
# Lagging with 1 period:
Func_Main_FitTable(_results_All_ts, ResultTable_All_ts)


## Since a regressor is removed from regression when it is duplicate with regressand, the OLS result of regression has to be adjusted in the result table to correctly reflect its effect.
def Func_Main_AdjustTable(Table, n):
    ## Check which regressand whose regression has removed a regressor as it is duplicate with the regressand and find out its order (column) in the result table.
    for col in range( len (TrueRegSpace[n] ) ):
        ## Once a regression has removed a regressor, its result (shown in column) has to be adjusted for the result (row) position for correctly evacuation of the regressor removed (shown in row).
        # i.e. the adjustment involves 2 steps: 1. evacuate the result of the regressor removed, 2. shift the results of regressors whose positions are after (below) the regressor removed to their corrsponding rows in the table.
        if len( TrueRegSpace[n][col] ) < len( TrueRegressors[n] ) + 1:         # '+1' since TrueRegSpace[X_order][col] including y.
            ## Calculate the order of the regressor removed in the regression.
            x_order = TrueRegressors[n].index( TrueRegSpace[n][col][0] )
            ## To preserve the info, the 2nd step would be executed first, and it is executed only when the regressor removed is not the last regressor in regression.
            # i.e. when the regressor removed is the last regressor in regression, there is no info of regressor whose position is after the regressor removed to be shifted.
            if x_order + 1 < len( TrueRegressors[n] ):
                for row in range( len( TrueRegressors[n] ) - x_order - 1):
                    Table.iloc[ (len( TrueRegressors[n] ) - 1) * 2 - row*2 , col + 1 ] = Table.iloc[ (len( TrueRegressors[n] ) - 1) * 2 - row*2 - 2 , col + 1 ]
                    Table.iloc[ (len( TrueRegressors[n] ) - 1) * 2 - row*2 + 1, col + 1 ] = Table.iloc[ (len( TrueRegressors[n] ) - 1) * 2 - row*2 - 2 + 1, col + 1 ]
             
            ## Execute the 1st step: evacuate the result of the regressor removed.
            Table.iloc[ x_order * 2 , col + 1 ] = np.nan
            Table.iloc[ x_order * 2 + 1 , col + 1 ] = np.nan

## Execute the adjustment of tables for each set of regressors.
for n in range( len( ResultTable_All ) ):
    Func_Main_AdjustTable( ResultTable_All[n], n )
    Func_Main_AdjustTable( ResultTable_Manufacturing[n], n )
    Func_Main_AdjustTable( ResultTable_NonManufacturing[n], n )
    Func_Main_AdjustTable( ResultTable_Services[n], n )
    #Func_Main_AdjustTable( ResultTable_Autor[n], n )
    for num in range( len(_results_Autor6) ):
        Func_Main_AdjustTable( ResultTables_Autor6[num][n], n )
    # Lagging with 1 period:
    Func_Main_AdjustTable( ResultTable_All_ts[n], n )