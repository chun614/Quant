" The following program is used to show the data processing procedures toward Taiwan census survey data. See 'WorkingPaper_Conc.pdf', which is available on the application platform, for details of further explanations on the data description and the data processing." 
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



### Data Access
## Import year 100 data.
_Subfolder = '.\Data\Survey\\'
_FileName = '100Survey.csv'
Table100 = pd.read_csv(_Subfolder + _FileName, index_col = False, low_memory = False, encoding='cp1252')

## Import year 95 data.
_FileName = '95Survey.csv'
Table95 = pd.read_csv(_Subfolder + _FileName, low_memory = False, encoding='cp1252')

## Import year 90 data.
_FileName = '90Survey_Manufacturing.xlsx'
Table90_M = pd.read_excel(_Subfolder + _FileName)

_FileName = '90Survey_Construction.xlsx'
Table90_C = pd.read_excel(_Subfolder + _FileName)

_FileName = '90Survey_Business.xlsx'
Table90_B = pd.read_excel(_Subfolder + _FileName)

_FileName = '90Survey_Logistic&Info.xlsx'
Table90_L = pd.read_excel(_Subfolder + _FileName)

_FileName = '90Survey_Finance.xlsx'
Table90_F = pd.read_excel(_Subfolder + _FileName)

_FileName = '90Survey_Service.xlsx'
Table90_S = pd.read_excel(_Subfolder + _FileName)

_FileName = 'IND90X.xlsx'
Table_90Categ = pd.read_excel(_Subfolder + _FileName, header = None)


## Convert the Chinese/Code labels to English.
# year 100 survey data
Table100.columns = ['Type of table', 'Level of business unit', 'Industrial code (major)', 'Industrial code (minor)', 'Type of organization', 'Opening year', 'Opening month', 'Main operating way (manufacturing)', 'Employee (male)', 'Employee (female)', 'Annual wage (employee)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Expenses', 'Sales', 'Income', 'Total asset', 'IT dummy', 'IT management', 'Info (IT)', 'Brand dummy', 'R&D dummy', 'Production amount', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Capital used', 'Fixed capital used', 'Amortizing (convenience shop)', 'Problem identified', 'Serial number']

# year 95 survey data
Table95.columns = ['Type of table', 'Level of business unit', 'Industrial code (major)', 'Industrial code (minor)', 'Type of organization', 'Opening year', 'Opening month', 'Second business dummy', 'Main operating way (manufacturing)', 'IT dummy', 'IT management', 'Info (IT)', 'Area (land)', 'Area (building)', 'Employee (male)', 'Employee (female)', 'Annual wage (employee)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Expenses', 'Income', 'Brand dummy', 'Total asset', 'R&D dummy', 'Production amount', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Capital used', 'Fixed capital used', 'Amortizing (convenience shop)', 'Serial number']

# year 90 survey data
Table90_M.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'Main operating way (manufacturing)', 'Triangular trade', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Supervisor & professional (male)', 'Supervisor & professional (female)', 'Annual wage (supervisor & professional)', 'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 'Salary to outsourcing', 'Labor (total)', 'Annual wage (total)', 'Raw material expense', 'Fuel expense', 'Electricity expense', 'Outsourcing expense', 'WIP&FPI (beginning)', 'WIP&FPI (ending)', 'COGP (annual)', 'COGS', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Sales', 'Domestic sales', 'Foreign sales', 'Income from repairment', 'Income from processing', 'Operating income', 'Income', 'Other non-operating income', 'Raw material used', 'WIP&FPI used', 'Current asset used', 'Land used (value)', 'Building used (value)', 'Pit used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Green expense', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Expenses on fuel, material, & electricity', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]

Table90_C.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Value of construction (annual)', 'Value of material (annual)', 'Supervisor (male)', 'Supervisor (female)', 'Annual wage (supervisor)', 'Employee: non-supervisor (male)', 'Employee: non-supervisor (female)', 'Annual wage (non-supervisor)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Raw material expense', 'Outsourcing expense', 'WIP&FPI (beginning)', 'WIP&FPI (ending)', 'Adjustment term (upside)', 'Land (expense)', 'Adjustment term (downside)', 'COGS', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Energy (expense)', 'Lease (expense)', 'Other expenses on construction site', 'Sales', 'Cost of Construction', 'Operating income', 'Income', 'Other non-operating income', 'Raw material used', 'WIP&FPI used', 'Land used (construction)', 'Current asset used', 'Land used (non-construction)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Green expense', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Expenses on fuel, material, & electricity', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]

Table90_B.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'Main operating way (business)', 'Triangular trade', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Supervisor & professional (male)', 'Supervisor & professional (female)', 'Annual wage (supervisor & professional)', 'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Raw material & fuel expense', 'WIP&FPI (beginning)', 'COGP (annual)', 'WIP&FPI (ending)', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Commission', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Product sales', 'Income from service', 'Income from catering', 'Income from commission', 'Operating income', 'Income', 'Other non-operating income', 'Raw material & WIP&FPI used', 'Current asset used', 'Land used (value)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Expenses on fuel, material, & electricity', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]

Table90_L.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Supervisor & professional (male)', 'Supervisor & professional (female)', 'Annual wage (supervisor & professional)', 'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Raw material expense', 'Fuel expense', 'Electricity expense', 'Expense on leasing transportation equipment', 'Shipping fee', 'Repairment expense', 'COGS', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Commission', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Passenger revenue', 'Cargo revenue', 'Warehouse revenue', 'Custom clearance revenue', 'Income from commission', 'Operating income', 'Income', 'Other non-operating income', 'Raw material & WIP&FPI used', 'Current asset used', 'Land used (value)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Expenses on fuel, material, & electricity', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]

Table90_F.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Supervisor & professional (male)', 'Supervisor & professional (female)', 'Annual wage (supervisor & professional)', 'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Interest expense on deposit', 'Other interest expense', 'Indemnity expense', 'Reserve', 'Re-insurance expense', 'Service charge', 'COGS', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Commission', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Interest income on loan', 'Other interest income', 'Income from service', 'Income from insurance premium', 'Income from investment', 'Operating income', 'Income', 'Other non-operating income', 'Raw material & WIP&FPI used', 'Security & ST asset used', 'Current asset used', 'Land used (value)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Monetary asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]

Table90_S.columns = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 'Opening year', 'Opening month', 'Type of organization', 'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 'Area (land)', 'Area (building)', 'Supervisor & professional (male)', 'Supervisor & professional (female)', 'Annual wage (supervisor & professional)', 'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 'Labor (total)', 'Annual wage (total)', 'Raw material & fuel expense', 'Service expense', 'Expense on leasing', 'COGS', 'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Commission', 'Operating expense', 'Expenses', 'Other non-operating expense', 'Other operating expense', 'Service revenue', 'Product sales', 'Income from real estate', 'Income from catering', 'Income from commission', 'Operating income', 'Income', 'Other non-operating income', 'Raw material & WIP&FPI used', 'Current asset used', 'Land used (value)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 'Production amount', 'Cost of production', 'Transfer & other expense', 'Expenses on fuel, material, & electricity', 'Gross production', 'Net production (mkt price)', 'Net production (cost)', 'Expense on leasing (Net amount)', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate'
    ]



### Extract-Transform-Load
## Check the nullity of main data.
# EDA: nullity patterns.
msno.matrix(Table100.sample(500))
msno.matrix(Table95.sample(500))
msno.matrix(Table90.sample(500))
msno.matrix(TablePI.iloc[:, 4:-1])

# EDA: simple visualization of nullity by column.
msno.bar(Table100.sample(1000))
msno.bar(Table95.sample(1000))
msno.bar(Table90.sample(1000))

# EDA: correlation of nullity among data.
# How strongly the presence or absence of one variable affects the presence of another.
msno.heatmap(Table100)
msno.heatmap(Table95)
msno.heatmap(Table90)

# ETL w.r.t. raw data.
df_100 = deepcopy(Table100)
df_95  = deepcopy(Table95)
df_90M = deepcopy(Table90_M)
df_90C = deepcopy(Table90_C)
df_90B = deepcopy(Table90_B)
df_90L = deepcopy(Table90_L)
df_90F = deepcopy(Table90_F)
df_90S = deepcopy(Table90_S)
df_90Categ = Table_90Categ.copy()


## Transform the data format to correct DataFrame setting.
for i in range(len(df_90Categ)):
    df_90Categ.loc[i, 'Code']     = re.sub('[ \u3000]', '', df_90Categ.loc[i, 0])[:4]
    df_90Categ.loc[i, 'Industry'] = re.sub('[ \u3000]', '', df_90Categ.loc[i, 0])[4:]

df_90Categ = df_90Categ.iloc[:, 1:]


## Industry Name -> Industrial Code.
def Func_IndCode(DGBAS, Categ):
    DGBAS['Industrial code (major)'] = ''
    
    for rowCateg in range(len(Categ)):
        DGBAS.loc[:, 'Industrial code (major)'][DGBAS['Industry'] == Categ['Industry'][rowCateg]] = Categ.iloc[rowCateg, 0]
    
    DGBAS['Industrial code (major)'] = DGBAS['Industrial code (major)'].apply(lambda x: int(x))


## Level of business -> Dummy
def Func_LvBusin(DGBAS):
    DGBAS['Level of business unit'] = ''
    
    DGBAS.loc[:, 'Level of business unit'][DGBAS['Level of business'] == '為獨立經營單位，一個場所即為一個企業。'] = 1
    DGBAS.loc[:, 'Level of business unit'][DGBAS['Level of business'] == '為分支單位，係指企業設置在不同據點營業之個別場所單位。'] = 2
    DGBAS.loc[:, 'Level of business unit'][DGBAS['Level of business'] == '為企業之總管理單位，統籌負責經營管理，且以該場所申報「營利事業所得稅」。'] = 3
    DGBAS.loc[:, 'Level of business unit'][DGBAS['Level of business'] == '為結合總管理單位及各分支單位而成的企業總稱。'] = 8
    
    DGBAS['Level of business unit'] = DGBAS['Level of business unit'].apply(lambda x: int(x))


## Choose the columns among the 6 categories of year 90 survey data for the sake of combining. (i.e. Common items among the 6 categories in year 90 data.)
columnName = ['Location', 'Level of business', 'Type of table', 'Industry', 'Serial number', 
              'Opening year', 'Opening month', 'Type of organization', 
              'IT dummy', 'E-business', 'Info (IT)', 'Online booking', 'E-transaction', 'Other use on e-business', 
              'Area (land)', 'Area (building)', 
              'Supervisor (male)', 'Supervisor (female)', 'Annual wage (supervisor)',
              'Employee: non-supervisor/professional (male)', 'Employee: non-supervisor/professional (female)', 
              'Annual wage (non-supervisor/professional)', 'Num of partners', 'Payroll of partners', 
              'Labor (total)', 'Annual wage (total)', 
              'Salary & subsidies', 'Depreciation', 'Tax & fees', 'Bad debt & transfer', 'Operating expense', 'Expenses', 
              'Other non-operating expense', 'Other operating expense', 'Operating income', 'Income', 
              'Other non-operating income', 'Current asset used', 
              'Land used (value)', 'Building used (value)', 'Transportation equipment used', 'Machines used', 
              'Construction work in progress & Prepaid machines used', 'Other asset used', 'Total asset', 
              'Fixed asset leased from others', 'Fixed asset leased to others', 'Expense on R&D & purchase of technology', 
              'Production amount', 'Cost of production', 'Transfer & other expense', 
              'Gross production', 'Net production (mkt price)', 'Net production (cost)', 
              'Expense on leasing', 'Interest expense', 'Profit', 'Capital used', 'Fixed capital used', 'Liquid asset', 
              'Net fixed asset owned', 'Gross fixed asset owned', 'Rate of value adding', 'Profit rate',
              'Industrial code (major)', 'Level of business unit']


df_90 = pd.DataFrame()
## Combine the 6 categories of year 90 survey data as one. Reset the index.
for _data in [df_90M, df_90C, df_90B, df_90L, df_90F, df_90S]:
    Func_IndCode(_data, df_90Categ)
    Func_LvBusin(_data)
    _data = _data[columnName]
    ## Unify column names of all of 6 categories of year 90 survey data.
    _data.columns = columnName
    df_90 = df_90.append(_data)

df_90 = df_90.reset_index(drop=True)



### Data Cleaning
## 1. Drop the observations whose 'Problem identified' column element is * in year 100 data.
df_100 = df_100.drop(df_100[df_100['Problem identified'] == '*'].index, axis = 0)

## 2. Drop the observations whose "level of business unit" dummy equals to 2.
# Since the observations are duplicated when their "Level of business unit" dummy equals 2 (i.e. branches), they are removed.
df_100 = df_100.drop(df_100[df_100['Level of business unit'] == 2].index, axis = 0)
df_95 = df_95.drop(df_95[df_95['Level of business unit'] == 2].index, axis = 0)
df_90 = df_90.drop(df_90[df_90['Level of business unit'] == 2].index, axis = 0)

## 3. Drop the observations whose "level of business unit" dummy equals to 3.
# Since the observations are duplicated when their "Level of business unit" dummy equals 3 (i.e. headquarters), they are removed.
df_100 = df_100.drop(df_100[df_100['Level of business unit'] == 3].index, axis = 0)
df_95 = df_95.drop(df_95[df_95['Level of business unit'] == 3].index, axis = 0)
df_90 = df_90.drop(df_90[df_90['Level of business unit'] == 3].index, axis = 0)

## 4. Drop the observations whose total asset are not strictly positive.
df_100 = df_100.drop(df_100[ df_100['Total asset'] <= 0 ].index, axis = 0)
df_95 = df_95.drop(df_95[ df_95['Total asset'] <= 0 ].index, axis = 0)
df_90 = df_90.drop(df_90[ df_90['Total asset'] <= 0 ].index, axis = 0)

## 5. Drop the observations whose fixed capital used are not strictly positive.
df_100 = df_100.drop(df_100[ df_100['Fixed capital used'] <= 0 ].index, axis = 0)
df_95 = df_95.drop(df_95[ df_95['Fixed capital used'] <= 0 ].index, axis = 0)
df_90 = df_90.drop(df_90[ df_90['Fixed capital used'] <= 0 ].index, axis = 0)



### Data Preprocess - Variable Selection and Generation
## Build up R&D dummy as there is no data about R&D expense amount in year 95 survey.
df_90['R&D dummy'] = ''
df_90.loc[:, 'R&D dummy'][ df_90['Expense on R&D & purchase of technology'] > 0 ] = 1
df_90.loc[:, 'R&D dummy'][ df_90['Expense on R&D & purchase of technology'] == 0 ] = 0
df_90['R&D dummy'] = df_90['R&D dummy'].apply(lambda x: int(x))

df_95['R&D dummy'] = df_95['R&D dummy'].apply(lambda x: 2-x)
df_100['R&D dummy'] = df_100['R&D dummy'].apply(lambda x: 2-x)

## Add up both gender of employees.
df_90['Employee (total)'] = df_90['Employee: non-supervisor/professional (male)'] + df_90['Employee: non-supervisor/professional (female)'] + df_90['Supervisor (male)'] + df_90['Supervisor (female)']
df_95['Employee (total)'] = df_95['Employee (male)'] + df_95['Employee (female)']
df_100['Employee (total)'] = df_100['Employee (male)'] + df_100['Employee (female)']

## Add up the wages of non-supervisor/professional employees and the supervisor/professional employees.
df_90['Annual wage (employee)'] = df_90['Annual wage (non-supervisor/professional)'] + df_90['Annual wage (supervisor)']


## Redefine the dummies with 0 as 'No' and 1 as 'Yes'.
# Cleaned in Saved Data.
# IT
df_90.loc[:, 'IT dummy'][ df_90['IT dummy'] == '無' ] = 0
df_90.loc[:, 'IT dummy'][ df_90['IT dummy'] == '有' ] = 1
df_90['IT dummy'] = df_90['IT dummy'].apply(lambda x: int(x))

df_90.loc[:, 'Info (IT)'][ df_90['Info (IT)'] == '無' ] = 0
df_90.loc[:, 'Info (IT)'][ df_90['Info (IT)'] == '有' ] = 1
df_90['Info (IT)'] = df_90['Info (IT)'].apply(lambda x: int(x))

df_95['IT dummy'] = df_95['IT dummy'].apply(lambda x: 2-x)
df_95['Info (IT)'] = df_95['Info (IT)'].apply(lambda x: 2-x)

df_100['IT dummy'] = df_100['IT dummy'].apply(lambda x: 2-x)
df_100.loc[:, 'Info (IT)'][ df_100['Info (IT)'] == 2 ] = 0


## Calculate the profit.
df_90['Profit'] = df_90['Income'] - df_90['Expenses']
df_95['Profit'] = df_95['Income'] - df_95['Expenses']
df_100['Profit'] = df_100['Income'] - df_100['Expenses']


## Retain only columns below: Industrial code, Labor (both employees and non-hiring staffs), Wage, Expense & Renvenue, R&D dummy, Net production (mkt price), Fixed Capital, Price index, IT.
Firm_90 = df_90[['Industrial code (major)', 'Labor (total)', 'Annual wage (total)', 'Profit', 'Expenses', 'Income', 'Total asset', 'R&D dummy', 'Net production (mkt price)', 'Fixed capital used', 'Price index', 'Employee (total)', 'Num of partners', 'Annual wage (employee)', 'Payroll of partners', 'IT dummy', 'Info (IT)']]
Firm_95 = df_95[['Industrial code (major)', 'Labor (total)', 'Annual wage (total)', 'Profit', 'Expenses', 'Income', 'Total asset', 'R&D dummy', 'Net production (mkt price)', 'Fixed capital used', 'Price index', 'Employee (total)', 'Num of partners', 'Annual wage (employee)', 'Payroll of partners', 'IT dummy', 'Info (IT)']]
Firm_100 = df_100[['Industrial code (major)', 'Labor (total)', 'Annual wage (total)', 'Profit', 'Expenses', 'Income', 'Total asset', 'R&D dummy', 'Net production (mkt price)', 'Fixed capital used', 'Price index', 'Employee (total)', 'Num of partners', 'Annual wage (employee)', 'Payroll of partners', 'IT dummy', 'Info (IT)']]

## Rename the columns for the sake of easy calling.
Firm_90.columns = ['Code', 'Labor', 'Wage', 'Profit', 'Expense', 'Inc', 'Firm size', 'R&D', 'Net prod', 'Fixed K', 'PI', 'Employees', 'Non-hiring staffs', 'Wage (employee)', 'Payroll (Non-hiring staffs)', 'IT', 'Info (IT)']
Firm_95.columns = ['Code', 'Labor', 'Wage', 'Profit', 'Expense', 'Inc', 'Firm size', 'R&D', 'Net prod', 'Fixed K', 'PI', 'Employees', 'Non-hiring staffs', 'Wage (employee)', 'Payroll (Non-hiring staffs)', 'IT', 'Info (IT)']
Firm_100.columns = ['Code', 'Labor', 'Wage', 'Profit', 'Expense', 'Inc', 'Firm size', 'R&D', 'Net prod', 'Fixed K', 'PI', 'Employees', 'Non-hiring staffs', 'Wage (employee)', 'Payroll (Non-hiring staffs)', 'IT', 'Info (IT)']



### Data Preprocess - Industrialization
## Calculate the industrial sums of different variables of interest.
def Func_Industrialize(Firm_data):
    Ind_data = Firm_data.groupby(['Code']).agg({'Wage'                        : 'sum',
                                                'Profit'                      : 'sum',
                                                'Net prod'                    : 'sum',
                                                'Fixed K'                     : 'sum',
                                                'Expense'                     : 'sum',
                                                'Inc'                         : 'sum',
                                                'PI'                          : 'mean',
                                                'Firm size'                   : 'sum',
                                                'R&D'                         : 'mean',
                                                'Labor'                       : 'sum',
                                                'IT'                          : 'mean',
                                                'Info (IT)'                   : 'mean',
                                                'Employees'                   : 'sum',
                                                'Non-hiring staffs'           : 'sum',
                                                'Wage (employee)'             : 'sum',
                                                'Payroll (Non-hiring staffs)' : 'sum'
                                                })
    Ind_data.columns = ['N.Wage', 'N.Profit', 'N.Net prod', 'N.Fixed K', 'N.Expense', 'N.Inc', 'PI', 'Industry size', 'R&D', 'Labor', 'IT', 'Info (IT)', 'Employees', 'Non-hiring staffs', 'N.Wage (employee)', 'N.Payroll (Non-hiring staffs)']
    
    ## Add column: number of firms for each industry.
    Ind_data['Firm Num'] = Firm_data.groupby(['Code']).count().iloc[:, 0]


    ## Calculate the real term of variables.
    Ind_data['R.Wage']     = Ind_data['N.Wage']      / Ind_data['PI']
    Ind_data['R.Profit']   = Ind_data['N.Profit']    / Ind_data['PI']
    Ind_data['R.Net prod'] = Ind_data['N.Net prod']  / Ind_data['PI']
    Ind_data['R.Fixed K']  = Ind_data['N.Fixed K']   / Ind_data['PI']
    Ind_data['R.Expense']  = Ind_data['N.Expense']   / Ind_data['PI']
    Ind_data['R.Inc']      = Ind_data['N.Inc']       / Ind_data['PI']

    ## Calculate the average term of variables.
    Ind_data['Avg Wage (per L)']         = Ind_data['R.Wage']        / Ind_data['Labor']
    Ind_data['N.Avg Wage (per L)']       = Ind_data['N.Wage']        / Ind_data['Labor']
    Ind_data['Avg Inc (per L)']          = Ind_data['R.Inc']         / Ind_data['Labor']
    Ind_data['N.Avg Inc (per L)']        = Ind_data['N.Inc']         / Ind_data['Labor']
    Ind_data['Avg Profit (per L)']       = Ind_data['R.Profit']      / Ind_data['Labor']
    Ind_data['N.Avg Profit (per L)']     = Ind_data['N.Profit']      / Ind_data['Labor']
    Ind_data['Avg Output (per L)']       = Ind_data['R.Net prod']    / Ind_data['Labor']
    Ind_data['N.Avg Output (per L)']     = Ind_data['N.Net prod']    / Ind_data['Labor']
    
    Ind_data['Avg Inc (per K)']          = Ind_data['R.Inc']         / Ind_data['R.Fixed K']
    Ind_data['Avg Profit (per K)']       = Ind_data['R.Profit']      / Ind_data['R.Fixed K']
    Ind_data['Avg Output (per K)']       = Ind_data['R.Net prod']    / Ind_data['R.Fixed K']
    
    Ind_data['Avg Inc (per firm)']       = Ind_data['R.Inc']         / Ind_data['Firm Num']
    Ind_data['Avg Profit (per firm)']    = Ind_data['R.Profit']      / Ind_data['Firm Num']
    Ind_data['Avg Output (per firm)']    = Ind_data['R.Net prod']    / Ind_data['Firm Num']
    Ind_data['Avg Labor (per firm)']     = Ind_data['Labor']         / Ind_data['Firm Num']
    Ind_data['Avg Capital (per firm)']   = Ind_data['R.Fixed K']     / Ind_data['Firm Num']
    Ind_data['Avg Firm size']            = Ind_data['Industry size'] / Ind_data['Firm Num']
    
    ## Calculate shares of variables.
    Ind_data['Wage share'] = Ind_data['R.Wage'] / Ind_data['R.Inc']

    Ind_data['Profit rate'] = Ind_data['R.Profit']  / Ind_data['R.Inc']

    ## Capital-Labor ratio
    Ind_data['N.K/L ratio'] = Ind_data['N.Fixed K'] / Ind_data['Labor']
    Ind_data['R.K/L ratio'] = Ind_data['R.Fixed K'] / Ind_data['Labor']


    ## Calculate concentration measures. (Based on income)
    # Group the income numbers based on industrial codes.
    I_IndInc = [list(i) for i in list(Firm_data.groupby(['Code'])['Inc'])]
    # -$$$ CHECK POINT $$$- Make sure that the matching of CR4/HHI to industry is based on industrial code exactly.
    #list(Ind_data.index) == [I_IndInc[i][0] for i in range(len(I_IndInc))]
    
    ## 1. Calculate concentration ratio for every industry as the ratio (sum of highest values of income : total income of the industry).
    def Func_CR(num, Ind_data):
        ## Create the column for the concentration ratio.
        colName = 'CR' + str(num)
        Ind_data[colName] = ''
        
        ## Reset index in order to correctly count the index when calculating the concentration ratio.
        Ind_data = Ind_data.reset_index()
        
        ## Start to calculate the concentration ratio.
        for i_CodeAmount in range(len(Ind_data)):
            Ind_data.loc[i_CodeAmount, colName] = np.sort(I_IndInc[i_CodeAmount][1])[-num:].sum() / I_IndInc[i_CodeAmount][1].sum()
        # Change to percentage by multiplying 100.
        Ind_data[colName] = Ind_data[colName].apply(lambda x: eval(str(x)) * 100)
        
        ## Set back the industrial code as the index of current dataframe.
        Ind_data = Ind_data.set_index('Code')
    
        return Ind_data
    
    # Calculate CR4; i.e., sum of highest 4 value of income : total income of the industry.
    Ind_data = Func_CR(4, Ind_data)
    # Ind_data = Func_CR(20, Ind_data)
    # -$$$ NOTE $$$- If add concentration ratio other than CR4, it has then to adjust elements of all column sorting in following code and the external document, Data Description.
    
    ## 2. Calculate HHI. (Based on income)
    Ind_data['HHI'] = ''
    for i_CodeAmount in range(len(Ind_data)):
        Ind_data.iloc[i_CodeAmount, -1] = ( ( I_IndInc[i_CodeAmount][1] / I_IndInc[i_CodeAmount][1].sum() )**2 ).sum()
    # As convention, the HHI should be mulitplied by 10000.
    Ind_data['HHI'] = Ind_data['HHI'].apply(lambda x: eval(str(x)) * 10000)
    
    return Ind_data

Ind_90  = Func_Industrialize(Firm_90)
Ind_95  = Func_Industrialize(Firm_95)
Ind_100 = Func_Industrialize(Firm_100)

SameInd_90  = deepcopy(Ind_90)
SameInd_95  = deepcopy(Ind_95)
SameInd_100 = deepcopy(Ind_100)


## Group the industries which do not appear in any of year 90, year 95, or year 100 data.
I_DiffInd = (SameInd_90.index | SameInd_95.index | SameInd_100.index) ^ (SameInd_90.index & SameInd_95.index & SameInd_100.index)

## In order to form balanced panel, drop the industries which do not appear in all of year 90, year 95, and year 100 data.
for i in range(len(I_DiffInd)):
    SameInd_90 = SameInd_90.drop(SameInd_90[ SameInd_90.index == I_DiffInd[i] ].index, axis = 0)
    SameInd_95 = SameInd_95.drop(SameInd_95[ SameInd_95.index == I_DiffInd[i] ].index, axis = 0)
    SameInd_100 = SameInd_100.drop(SameInd_100[ SameInd_100.index == I_DiffInd[i] ].index, axis = 0)



### Exploratory data analysis
 ### CR4: Income vs Sales
## For year 100 data, compare the CR4 & HHI between based on income and based on sales.
# Group the sales numbers based on industrial code.
EDA_Var_100 = df_100[['Industrial code (major)', 'Labor (total)', 'Annual wage (total)', 'Expenses', 'Income', 'Total asset', 'R&D dummy', 'Net production (mkt price)', 'Fixed capital used', 'Price index', 'Employee (total)', 'Num of partners', 'Sales']]
EDA_Var_100 = EDA_Var_100.drop(EDA_Var_100[ (EDA_Var_100['Expenses'] <= 0) & (EDA_Var_100['Income'] <= 0) ].index, axis = 0)
EDA_IndSales = [list(i) for i in list(EDA_Var_100.groupby(['Industrial code (major)'])['Sales'])]

## Compare the CR4 & HHI between based on income and sales income.
# Show the heatmap of their correlations.
EDA_Corr = Ind_100[['CR4','CR4_Sales', 'HHI', 'HHI_Sales']].corr()
sns.heatmap(EDA_Corr,cmap="BrBG",annot=True)
EDA_Corr


 ### Box plot
sns.boxplot(x=Ind_95['R.Profit'])
#plt.boxplot(Ind_95['R.Profit'])
#Ind_95[['R.Profit', 'R.Inc']].boxplot()
plt.show()


 ### Scatter
# Relationship between profit and income.
# Inc vs Profit (Industrial)
arg_fig, arg_ax = plt.subplots(figsize=(10,6))
arg_ax.scatter(Ind_95['R.Inc']/1000, Ind_95['R.Profit']/1000)
arg_ax.set_xlabel('Inc')
arg_ax.set_ylabel('Profit')
plt.show()

# Inc vs Profit (Firm)
arg_fig, arg_ax = plt.subplots(figsize=(10,6))
sns.regplot(x='Inc',y= 'Profit', data=Var_95, ax=arg_ax)
plt.show()

# Industrial
sns.set()
sns.pairplot(Ind_95[['Profit rate', 'Avg Labor (per firm)', 'Avg Capital (per firm)', 'K/L ratio', 'Wage share', 
                        'Avg Profit (per L)', 'Avg Output (per K)', 'CR4', 'HHI']], height = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


 ### Heatmap
## Correlation among industrial data.
plt.figure(figsize=(10,5))
EDA_Corr = Ind_95[['R.Profit', 'R.Net prod', 'R.Expense', 'R.Inc', 'CR4', 'HHI', 
               'Avg Output (per L)', 'Wage share', 'K/L ratio']].corr()
sns.heatmap(EDA_Corr,cmap="BrBG",annot=True)
EDA_Corr

## Correlation among firm data.
plt.figure(figsize=(10,5))
EDA_Corr = Var_95[['Profit', 'Net prod', 'Expense', 'Inc']].corr()
sns.heatmap(EDA_Corr,cmap="BrBG",annot=True)
EDA_Corr
# -$$$ NOTE $$$- Heatmap color choices (cmap=):
# "YlGnBu" for cold color;
# "seismic" for blue to red;
# "PuOr" for brown to violet;
# "PRGn" for violet to green;
# "viridis" for violet, blue, green, yellow.

## Correlation among all industrial data.
EDA_Corr = Ind_95.corr()
#EDA_Corr = Ind_95.iloc[:, :25].corr()
plt.figure(figsize=(15, 18))
sns.heatmap(EDA_Corr[(EDA_Corr >= 0.5) | (EDA_Corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True,
            linecolor = 'red')

## Correlation among profit rate and other inputs of industry.
plt.figure(figsize=(10,5))
EDA_Corr = Ind_95[['Profit rate', 'Avg Labor (per firm)', 'Avg Capital (per firm)', 'K/L ratio', 'Wage share', 'Industry size',
                      'CR4', 'HHI']].corr()
sns.heatmap(EDA_Corr,cmap="seismic",annot=True)
EDA_Corr


 ### Category (Distribution)
## Look on distribution of profit.
# Industry profit.
EDA_cats = pd.cut( Ind_95['R.Profit'],
             [-10000000, -100000, -10000, -1000, -100, -10 ,0, 10, 100, 1000, 10000, 100000, 1000000, 10000000] )
sns.distplot(EDA_cats.cat.codes, color='g', hist_kws={'alpha': 0.5})
EDA_cats.value_counts().sort_index()

# Firm profit.
EDA_cats_firm = pd.cut( Var_95['Profit'] / Var_95['PI'],
             [-10000000, -100000, -10000, -1000, -100, -10 ,0, 10, 100, 1000, 10000, 100000, 1000000, 10000000] )
EDA_cats_firm.cat.codes.hist(figsize=(8, 10), bins=40, xlabelsize=13, ylabelsize=13)
EDA_cats_firm.value_counts().sort_index()


 ### Histogram (Distribution)
Ind_95['R.Profit'].hist(figsize=(8, 10), bins=80, xlabelsize=13, ylabelsize=13)

## Look on distribution of industrial profit.
plt.figure(figsize=[10,8])
arg_n, arg_bins, arg_patches = plt.hist(x=Ind_95['R.Profit'], bins=80, color='#0504aa',alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Profit',fontsize=15)
plt.ylabel('Amount of Industry',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Industrial Profit',fontsize=15)
plt.show()

plt.figure(3); plt.title('Log Normal')
sns.distplot(EDA_cats.cat.codes, kde=False, fit=st.lognorm)