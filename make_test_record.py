import numpy as np
import tensorflow as tf
import h5py
import re
from tqdm import tqdm
import sentencepiece as spm
import pickle

re_sc = re.compile(r'[\!@#$%♡★_\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

test_files=['../item_clf/train.chunk.06']

pids=[]
titles=[]
bcateids=[]
mcateids=[]
scateids=[]
dcateids=[]

for file in test_files:
    with h5py.File(file,'r') as f:
        data=f['train']
        pids+=data['pid']
        titles+=data['product']
        bcateids+=data['bcateid']
        mcateids+=data['mcateid']
        scateids+=data['scateid']
        dcateids+=[0 if elem<0 else elem for elem in data['dcateid']]

titles=[re_sc.sub(' ',elem.decode('utf-8')) for elem in titles]

sp=spm.SentencePieceProcessor()
sp.load('./vocab/spm.model')

titles_sp=[sp.EncodeAsIds(title) for title in titles]

data=[{'pid':z[0],'title':z[1],'bcateid':z[2],'mcateid':z[3],'scateid':z[4],'dcateid':z[5]}
        for z in zip(pids,titles_sp,bcateids,mcateids,scateids,dcateids)
        if z[4]>0 and len(z[1])>4 and len(z[1])<31]

with open('test.pkl','w') as f:
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)