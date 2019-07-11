import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,metrics,Model,optimizers,regularizers,Sequential,activations
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_pieces=5
num_tokens=7

def get_encoder(hidden_size,vocab_size,
                 num_tokens=7,nlayers=1,dropout=0.2,
                 bsize=57, msize=552, ssize=3190, dsize=404):
    len_input=keras.Input(shape=(),name='len',dtype=tf.int64)
    pieces_input=[keras.Input(shape=(num_pieces,),name='piece{}'.format(i+1)) for i in range(num_tokens)]

    embedding=layers.Embedding(vocab_size,200,mask_zero=True)
    pieces=[embedding(piece) for piece in pieces_input]
    cells=[layers.LSTMCell(hidden_size,dropout=dropout) for _ in range(nlayers-1)]
    cells.append(layers.LSTMCell(hidden_size))
    lstm=layers.RNN(cells,return_sequences=False,return_state=True,name='multi-lstm')



    state=lstm(pieces[0])[-1][-1]
    states=[state]

    pieces.remove(pieces[0])

    zero_state=tf.zeros_like(state)

    sent_len=tf.reshape(len_input,(-1,1))

    for i,piece in enumerate(pieces):
    # for piece in pieces:
        state=tf.where(i+1<sent_len,lstm(piece)[-1][-1],zero_state)
        # state= lstm(piece)[-1][-1]
        states.append(state)

    result=tf.math.add_n(states)
    
    #sent_len=tf.tile(len_input,[1,hidden_size])
    sent_len=tf.cast(sent_len,tf.float32)
    result=tf.divide(result,sent_len)

    feature=Sequential([
        layers.Dense(hidden_size,activations.relu),
        layers.Dropout(dropout),
        layers.Dense(hidden_size,activations.relu,name='final_feature')
    ],name='feature_seq')(result)

    bcate = layers.Dense(bsize,name='bcateid')(feature)
    mcate = layers.Dense(msize,name='mcateid')(feature)
    scate = layers.Dense(ssize,name='scateid')(feature)
    dcate = layers.Dense(dsize,name='dcateid')(feature)

    inputs=[len_input]+pieces_input

    model=Model(inputs=inputs,outputs=[bcate,mcate,scate,dcate])
    return model