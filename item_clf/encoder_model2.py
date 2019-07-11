import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,metrics,Model,optimizers,regularizers,Sequential,activations
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_pieces=5
num_tokens=7

def get_encoder(hidden_size,vocab_size,
                 num_tokens=7,nlayers=2,dropout=0.2,
                 bsize=57, msize=552, ssize=3190, dsize=404):
    len_input=keras.Input(shape=(),name='len',dtype=tf.int64)
    pieces_input=[keras.Input(shape=(num_pieces,),name='piece{}'.format(i+1)) for i in range(num_tokens)]
    img_input=keras.Input(shpae=(2048,),name='img')

    embedding=layers.Embedding(vocab_size,hidden_size,mask_zero=True)
    pieces=[embedding(piece) for piece in pieces_input]
    cells=[layers.GRUCell(hidden_size,dropout=dropout) for _ in range(nlayers-1)]
    cells.append(layers.LSTMCell(hidden_size))
    lstm=layers.RNN(cells,return_sequences=False,return_state=True,name='multi-gru')



    state=lstm(pieces[0])
    states=[state[-1][-1]]

    pieces.remove(pieces[0])

    for piece in pieces:
        state=lstm(piece)
        states.append(state[-1][-1])

    result=tf.math.add_n(states)
    sent_len=tf.reshape(len_input,(-1,1))
    #sent_len=tf.tile(len_input,[1,hidden_size])
    sent_len=tf.cast(sent_len,tf.float32)
    text_feat=tf.divide(result,sent_len,name='text_feature')

    img_feat=layers.Dense(hidden_size,activations.relu,name='img_feature')(img_input)

    text_plus_img=layers.concat([text_feat,img_feat],1)

    feature=Sequential([
        layers.Dense(hidden_size,activations.relu),
        layers.Dropout(dropout),
        layers.Dense(hidden_size,activations.relu,name='final_feature')
    ],name='feature_seq')(text_plus_img)

    bcate = layers.Dense(bsize,name='bcateid')(feature)
    mcate = layers.Dense(msize,name='mcateid')(feature)
    scate = layers.Dense(ssize,name='scateid')(feature)
    dcate = layers.Dense(dsize,name='dcateid')(feature)

    inputs=[len_input,img_input]+pieces_input

    model=Model(inputs=inputs,outputs=[bcate,mcate,scate,dcate])
    return model