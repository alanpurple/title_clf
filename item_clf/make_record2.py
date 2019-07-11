import numpy as np
import tensorflow as tf
import h5py
import re
from tqdm import tqdm
import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences

re_sc = re.compile(r'[\!@#$%♡★_\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

train_files=[
    'train.chunk.01','train.chunk.02','train.chunk.03','train.chunk.04','train.chunk.05','train.chunk.06',
    'train.chunk.07','train.chunk.08','train.chunk.09'
]
dev_files=['dev.chunk.01']

num_tokens=7
num_pieces=5

titles=[]
bcateids=[]
mcateids=[]
scateids=[]
dcateids=[]
img=[]

for file in train_files:
    with h5py.File(file,'r') as f:
        data=f['train']
        titles+=data['product']
        bcateids+=[0 if elem<0 else elem for elem in data['bcateid']]
        mcateids+=[0 if elem<0 else elem for elem in data['mcateid']]
        scateids+=[0 if elem<0 else elem for elem in data['scateid']]
        dcateids+=[0 if elem<0 else elem for elem in data['dcateid']]

titles_dev=[]
bcateids_dev=[]
mcateids_dev=[]
scateids_dev=[]
dcateids_dev=[]
img_dev=[]

for file in dev_files:
    with h5py.File(file,'r') as f:
        data=f['dev']
        titles_dev+=data['product']
        bcateids_dev+=[0 if elem<0 else elem for elem in data['bcateid']]
        mcateids_dev+=[0 if elem<0 else elem for elem in data['mcateid']]
        scateids_dev+=[0 if elem<0 else elem for elem in data['scateid']]
        dcateids_dev+=[0 if elem<0 else elem for elem in data['dcateid']]

title_split=[elem.decode('utf-8') for elem in titles]
title_split=[' '.join(re_sc.sub(' ',title).strip().split()) for title in tqdm(title_split,mininterval=1)]

title_dev_split=[elem.decode('utf-8') for elem in titles_dev]
title_dev_split=[' '.join(re_sc.sub(' ',title).strip().split()) for title in tqdm(title_dev_split,mininterval=1)]

with open('titles.txt','w') as f:
    for title in tqdm(title_split,mininterval=1):
        f.write(title+'\n')

spm.SentencePieceTrainer_Train('--input=titles.txt --model_type=bpe --model_prefix=./vocab/spm vocab_size=8000')

sp=spm.SentencePieceProcessor()
sp.load('./vocab/spm.model')

title_split_sp=[[sp.EncodeAsIds(elem) for elem in title] for title in title_split]

title_split_sp_dev=[[sp.EncodeAsIds(elem) for elem in title] for title in title_dev_split]

def _int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

with tf.io.TFRecordWriter('sp_category_train2.tfrecord') as writer:
    for j,title in enumerate(title_split_sp):
        feature={
            'bcateid':_int64_feature(bcateids[j]),
            'mcateid':_int64_feature(mcateids[j]),
            'scateid':_int64_feature(scateids[j]),
            'dcateid':_int64_feature(dcateids[j]),
            'len':_int64_feature(num_tokens)
            }
        if len(title)<num_tokens:
            feature['len']=_int64_feature(len(title))
        title=pad_sequences(title,num_pieces,padding='post', truncating='post')
        for i in range(num_tokens):
            if(len(title))<i+1:
                feature['piece{}'.format(i+1)]=_int64_list_feature([0]*num_pieces)
            else:
                feature['piece{}'.format(i+1)]=_int64_list_feature(title[i])
        example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
        example=example_proto.SerializeToString()
        writer.write(example)

with tf.io.TFRecordWriter('sp_category_dev2.tfrecord') as writer:
    for j,title in enumerate(title_split_sp_dev):
        feature={
            'bcateid':_int64_feature(bcateids_dev[j]),
            'mcateid':_int64_feature(mcateids_dev[j]),
            'scateid':_int64_feature(scateids_dev[j]),
            'dcateid':_int64_feature(dcateids_dev[j]),
            'len':_int64_feature(num_tokens)
            }
        if len(title)<num_tokens:
            feature['len']=_int64_feature(len(title))
        title=pad_sequences(title,num_pieces,padding='post', truncating='post')
        for i in range(num_tokens):
            if(len(title))<i+1:
                feature['piece{}'.format(i+1)]=_int64_list_feature([0]*num_pieces)
            else:
                feature['piece{}'.format(i+1)]=_int64_list_feature(title[i])
        example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
        example=example_proto.SerializeToString()
        writer.write(example)