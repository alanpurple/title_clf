import numpy as np
import tensorflow as tf
import h5py
import re
from tqdm import tqdm
import sentencepiece as spm
from tensorflow.keras.preprocessing.sequence import pad_sequences

re_sc = re.compile(r'[\!@#$%♡★_\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

train_files=[
    'train.chunk.01','train.chunk.02'
]

titles=[]
bcateids=[]
mcateids=[]
scateids=[]
dcateids=[]

for file in train_files:
    with h5py.File(file,'r') as f:
        data=f['train']
        titles+=data['product']
        bcateids+=data['bcateid']
        mcateids+=data['mcateid']
        scateids+=[0 if elem<0 else elem for elem in data['scateid']]
        dcateids+=[0 if elem<0 else elem for elem in data['dcateid']]

titles=[re_sc.sub(' ',elem.decode('utf-8')) for elem in titles]

with open('titles.txt','w') as f:
    for title in tqdm(titles,mininterval=1):
        f.write(title+'\n')

spm.SentencePieceTrainer_Train('--input=titles.txt --model_type=bpe --model_prefix=./vocab/spm vocab_size=8000')

sp=spm.SentencePieceProcessor()
sp.load('./vocab/spm.model')

titles_sp=[sp.EncodeAsIds(title) for title in titles]

def _int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

with tf.io.TFRecordWriter('sp_category_train.tfrecord') as writer:
    for j,title in enumerate(titles_sp):
        if len(title)>4 and len(title)<31:
            feature={
                'bcateid':_int64_feature(bcateids[j]),
                'mcateid':_int64_feature(mcateids[j]),
                'scateid':_int64_feature(scateids[j]),
                'dcateid':_int64_feature(dcateids[j])
                }
            if len(title)<20:
                title.extend([0]*(20-len(title)))
            else:
                title=title[:20]
            feature['pieces']=_int64_list_feature(title)
            example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
            example=example_proto.SerializeToString()
            writer.write(example)

print('done')