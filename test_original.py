import pandas as pd
import numpy as np
import pickle

df=pd.read_csv('dev.tsv',sep='\t',names=['bcateid','mcateid','scateid','dcateid'],index_col=0)

with open('test.pkl','rb') as f:
    data=pickle.load(f)

bcount=0
mcount=0
scount=0

for elem in data:
    temp=df.loc[elem['pid'].decode('utf-8')]
    if temp['bcateid']==elem['bcateid']:
        bcount+=1
    if temp['mcateid']==elem['mcateid']:
        mcount+=1
    if temp['scateid']==elem['scateid']:
        scount+=1

print('bcate: {} \n'.format(bcount))
print('mcate: {} \n'.format(mcount))
print('scate: {} \n'.format(scount))