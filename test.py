from tensorflow.keras import models,Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import re
from encoder_model import get_encoder
import sentencepiece as spm
import json

re_sc = re.compile(r'[\!@#$%♡★_\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')

titles=['[랭킹특가] 풋사과 다이어트젤리 1+1스키니랩/곤약/시서스/다이어트',
        '[AK몰] 나이키 에어맥스/탄준/에어포스/오디세이/코르테즈/레볼루션',
        'S329 저소음 벽시계 인테리어시계 벽걸이시계 생활소',
        '[하프클럽][아쿠아티카] 아쿠아티카 BH03 남자수영복 5부수영복']

titles=[re_sc.sub(' ',title) for title in tqdm(titles,mininterval=1)]

latest_ckpt=tf.train.latest_checkpoint('train10')

sp=spm.SentencePieceProcessor()
sp.load('./vocab/spm.model')

titles_sp=[sp.EncodeAsIds(title) for title in titles]

titles_sp=pad_sequences(titles_sp,20)

feat_desc={
    'pieces':tf.io.FixedLenFeature([20],tf.int64)
}

test_data={'pieces':titles_sp}

test_data=tf.data.Dataset.from_tensor_slices((test_data)).batch(2)
model=get_encoder(700,8000)
model.load_weights(latest_ckpt)

pred=model.predict(test_data)

with open('cate1.json','r') as f:
    cate_json=json.load(f)

bcates=cate_json['b']
mcates=cate_json['m']
scates=cate_json['s']
dcates=cate_json['d']

bdict={v:k for k,v in bcates.items()}
mdict={v:k for k,v in mcates.items()}
sdict={v:k for k,v in scates.items()}
ddict={v:k for k,v in dcates.items()}

bpred=np.argmax(pred[0],axis=1)
mpred=np.argmax(pred[1],axis=1)
spred=np.argmax(pred[2],axis=1)
dpred=np.argmax(pred[3],axis=1)

for i in range(len(titles)):
    if bpred[i] in bdict.keys():
        print('bcate: {}'.format(bdict[bpred[i]]))
    if mpred[i] in mdict.keys():
        print('mcate: {}'.format(mdict[mpred[i]]))
    if spred[i] in sdict.keys():
        print('scate: {}'.format(sdict[spred[i]]))
    if dpred[i] in ddict.keys():
        print('dcate: {}'.format(ddict[dpred[i]]))

    print('\n\n')

feature_layer=list(filter(lambda x:x.name=='feature_seq', model.layers))[0]

feature_ext_model=Model(inputs=model.input,outputs=feature_layer.output)

feature_ext_model.load_weights(latest_ckpt)

result=feature_ext_model.predict(test_data)

for elem in result:
    print(elem[:30])