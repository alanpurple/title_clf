import tensorflow as tf
import pickle
from tensorflow.keras import Model
from encoder_model2 import get_encoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

test_file='test.pkl'
latest_ckpt=tf.train.latest_checkpoint('train2')

with open(test_file,'rb') as f:
    data=pickle.load(f)

titles=[elem['title'] for elem in data]
titles=pad_sequences(titles,20)

test_data={'pieces':titles}

test_data=tf.data.Dataset.from_tensor_slices((test_data)).batch(25)

model=get_encoder(700,8000)
model.load_weights(latest_ckpt)

pred=model.predict(test_data)

bpred=np.argmax(pred[0],axis=1)
mpred=np.argmax(pred[1],axis=1)
spred=np.argmax(pred[2],axis=1)
# dpred=np.argmax(pred[3],axis=1)

bcount=0
mcount=0
scount=0

for i in range(len(titles)):
    if bpred[i]==data[i]['bcateid']:
        bcount+=1
    if mpred[i]==data[i]['mcateid']:
        mcount+=1
    if spred[i]==data[i]['scateid']:
        scount+=1

print('bcate: {} \n'.format(bcount))
print('mcate: {} \n'.format(mcount))
print('scate: {} \n'.format(scount))