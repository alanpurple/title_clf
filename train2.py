import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,metrics,Model,optimizers,regularizers,Sequential,activations
from tensorflow.keras.preprocessing.sequence import pad_sequences
from encoder_model2 import get_encoder

train_filenames=['sp_category_train.tfrecord']

def loss_fn(real ,pred):
    cce=losses.SparseCategoricalCrossentropy(True)
    return cce(real,pred)

# handling for unknown types
def loss_fn_many(real,pred):
    zero_tensor=tf.zeros_like(real)
    one_tensor=tf.ones_like(real)
    weights=tf.where(real>0,one_tensor,zero_tensor)
    cce=losses.SparseCategoricalCrossentropy(True)
    return cce(real,pred,weights)

def main():
    raw_dataset=tf.data.TFRecordDataset(train_filenames)
    raw_dataset=raw_dataset.take(720000)

    feat_desc={
        'bcateid':tf.io.FixedLenFeature([],tf.int64),
        'mcateid':tf.io.FixedLenFeature([],tf.int64),
        'scateid':tf.io.FixedLenFeature([],tf.int64),
        'dcateid':tf.io.FixedLenFeature([],tf.int64),
        'pieces':tf.io.FixedLenFeature([20],tf.int64)
    }

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
    dataset=dataset.batch(40,True)

    encoder=get_encoder(700,8000)

    keras.utils.plot_model(encoder,'text_clf_gru.png',show_shapes=True)
    
    encoder.compile(
        optimizer=optimizers.Adam(),
        loss={
            'bcateid':loss_fn,
            'mcateid':loss_fn,
            'scateid':loss_fn_many,
            'dcateid':loss_fn_many
        },
        metrics=[metrics.CategoricalAccuracy()])

    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath='train10/spmodel_{epoch}.ckpt',
            save_weights_only=True,
            verbose=1
        )
    ]

    encoder.fit(dataset,epochs=3,callbacks=callbacks)

if __name__=='__main__':
    main()