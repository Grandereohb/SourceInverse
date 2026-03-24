# coding: utf-8
import pandas as pd
site = pd.read_excel(r"C:\Document\博士\高老师项目\溯源\data_sumitomo\sites.xlsx")
conc = pd.read_excel(r"C:\Document\博士\高老师项目\溯源\data_sumitomo\厂界\test3\3.xlsx")
wind = pd.read_excel(r"C:\Document\博士\高老师项目\溯源\data_sumitomo\厂界\test3\wind.xlsx")
print("sites columns:", list(site.columns))
print(site.head())
print("conc columns:", list(conc.columns))
print(conc.head())
print("wind columns:", list(wind.columns))
print(wind.head())
