from tensorflow.keras import models,Model
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
import re
from encoder_model4 import get_encoder
import sentencepiece as spm

re_sc = re.compile(r'[\!@#$%♡★_\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

num_tokens=7
num_pieces=3

titles=['[랭킹특가] 풋사과 다이어트젤리 1+1스키니랩/곤약/시서스/다이어트',
        '[AK몰] 나이키 에어맥스/탄준/에어포스/오디세이/코르테즈/레볼루션',
        'S329 저소음 벽시계 인테리어시계 벽걸이시계 생활소',
        '[하프클럽][아쿠아티카] 아쿠아티카 BH03 남자수영복 5부수영복']

latest_ckpt=tf.train.latest_checkpoint('train')

title_split=[' '.join(re_sc.sub(' ',title).strip().split()) for title in tqdm(titles,mininterval=1)]
title_split=[title.split() for title in title_split]

sp=spm.SentencePieceProcessor()
sp.load('./vocab/spm.model')

title_split_sp=[[sp.EncodeAsIds(elem) for elem in title] for title in title_split]

feat_desc={
        'len':tf.io.FixedLenFeature([],tf.int64)
    }
for i in range(num_tokens):
    feat_desc['piece{}'.format(i+1)]=tf.io.FixedLenFeature([num_pieces],tf.int64)

test_data={'len':[]}
for i in range(num_tokens):
    test_data['piece{}'.format(i+1)]=[]
for title in title_split_sp:
    if len(title)<num_tokens:
            test_data['len'].append(len(title))
    else:
        test_data['len'].append(num_tokens)
    title=pad_sequences(title,num_pieces,padding='post', truncating='post')
    for i in range(num_tokens):
        if(len(title))<i+1:
            test_data['piece{}'.format(i+1)].append([0]*num_pieces)
        else:
            test_data['piece{}'.format(i+1)].append(title[i])

test_data=tf.data.Dataset.from_tensor_slices((test_data)).batch(2)
model=get_encoder(700,8000,num_tokens)
model.load_weights(latest_ckpt)

pred=model.predict(test_data)

for cate in pred:
    print(np.argmax(cate,axis=1))


feature_layer=list(filter(lambda x:x.name=='feature_seq', model.layers))[0]

feature_ext_model=Model(inputs=model.input,outputs=feature_layer.output)

feature_ext_model.load_weights(latest_ckpt)

result=feature_ext_model.predict(test_data)

for elem in result:
    print(elem[:30])