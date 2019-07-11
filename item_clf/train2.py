import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,metrics,Model,optimizers,regularizers,Sequential,activations
from tensorflow.keras.preprocessing.sequence import pad_sequences
from encoder_model import get_encoder

train_filenames=['sp_category_train.tfrecord']
num_pieces=5
num_tokens=7

def loss_fn(real ,pred):
    cce=losses.SparseCategoricalCrossentropy()
    return cce(real,pred)

# handling for unknown types
def loss_fn_many(real,pred):
    zero_tensor=tf.zeros_like(real)
    one_tensor=tf.ones_like(real)
    weights=tf.where(real>0,one_tensor,zero_tensor)
    cce=losses.SparseCategoricalCrossentropy()
    return cce(real,pred,weights)

# @tf.function
# def train_step(encoder,input,target):
#     loss=0
    
#     with tf.GradientTape() as tape:
#         b_cate,m_cate,s_cate,d_cate=encoder(input)



def main():
    raw_dataset=tf.data.TFRecordDataset(train_filenames)
    raw_dataset=raw_dataset.take(16000)

    feat_desc={
        'bcateid':tf.io.FixedLenFeature([],tf.int64),
        'mcateid':tf.io.FixedLenFeature([],tf.int64),
        'scateid':tf.io.FixedLenFeature([],tf.int64),
        'dcateid':tf.io.FixedLenFeature([],tf.int64),
        'len':tf.io.FixedLenFeature([],tf.int64),
        'img':tf.io.FixedLenFeature([2048],tf.float32)
    }
    for i in range(num_tokens):
        feat_desc['piece{}'.format(i+1)]=tf.io.FixedLenFeature([num_pieces],tf.int64)

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto,feat_desc)
        output={}
        output['bcateid']=parsed.pop('bcateid')
        output['mcateid']=parsed.pop('mcateid')
        output['scateid']=parsed.pop('scateid')
        output['dcateid']=parsed.pop('dcateid')
        return (parsed,output)

    dataset=raw_dataset.map(_parse_function)
    dataset=dataset.shuffle(1024)
    dataset=dataset.batch(20)

    # encoder=Encoder(200,24000,num_tokens)
    encoder=get_encoder(200,8000,num_tokens)

    keras.utils.plot_model(encoder,'enocoder2_model.png',show_shapes=True)

    encoder.compile(
        optimizer=optimizers.Adam(),
        loss={
            'bcateid':loss_fn,
            'mcateid':loss_fn,
            'scateid':loss_fn_many,
            'dcateid':loss_fn_many
        },
        metrics=[metrics.CategoricalCrossentropy()])

    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath='train/spmodel_{epoch}.ckpt',
            save_weights_only=True,
            verbose=1
        )
    ]

    encoder.fit(dataset,epochs=3,callbacks=callbacks)
    
    


if __name__=='__main__':
    main()