# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:19:02 2019

@author: amris
"""

import pandas as pn
import numpy as nm

D = pn.read_csv(r'C:\Users\amris\Downloads\mtcars.csv')
D.corr()
#corr will give values in heatmap n see which columns shouldbe taken in x and which column has highest corr value so will go in Y dependet column
import matplotlib.pyplot as pt
import seaborn as sbs

sbs.heatmap(D.corr(),annot=True)# to find the correlation of whole data if u put aanot= true u get values in Heatmap
D.columns
x= D.drop (['Unnamed: 0', 'mpg','drat', 'wt', 'qsec', 'vs','am', 'gear', 'carb'],axis=1)
y= pn.DataFrame(D.wt)
from sklearn.linear_model import LinearRegression
obj= LinearRegression()
obj.fit(x,y)
pr=obj.predict(x)

from sklearn.metrics import r2_score
acc= r2_score(y,pr)
print(acc*100)
#if mpg=36 and cyl=4 then predict carb value 

x=  D.drop (['Unnamed: 0', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs','am', 'gear', 'carb'],axis=1)
y= pn.DataFrame(D.carb)

from sklearn.linear_model import LinearRegression
obj= LinearRegression()
obj.fit(x,y)
pr=obj.predict(x)
b=obj.predict([[36,4]])#entred values of mpg and cyl and get carb predicted value

from sklearn.metrics import r2_score
acc= r2_score(y,pr)
print(acc*100)
if acc*100<60:
    print('prediction result wont be right')
