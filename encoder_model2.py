import tensorflow as tf
from tensorflow.keras import layers,Model,Input,Sequential,activations

def get_encoder(hidden_size,vocab_size,dropout=0.2,num_layers=2,
                bsize=58,msize=553,ssize=3191,dsize=405):
    pieces=Input(shape=(20,),name='pieces')

    embedding=layers.Embedding(vocab_size,200,mask_zero=True)
    pieces_emb=embedding(pieces)
    cells=[layers.LSTMCell(hidden_size,dropout=dropout) for _ in range(num_layers-1)]
    cells.append(layers.LSTMCell(hidden_size))
    lstm=layers.RNN(cells,return_sequences=False,return_state=True,name='multi-lstm')

    output=lstm(pieces_emb)[-1][-1]

    feature=Sequential([
        layers.Dense(hidden_size,activations.relu),
        layers.Dropout(dropout),
        layers.Dense(hidden_size,activations.relu,name='final_feature')
    ],name='feature_seq')(output)

    bcate = layers.Dense(bsize,name='bcateid')(feature)
    mcate = layers.Dense(msize,name='mcateid')(feature)
    scate = layers.Dense(ssize,name='scateid')(feature)
    dcate = layers.Dense(dsize,name='dcateid')(feature)

    model=Model(inputs=pieces,outputs=[bcate,mcate,scate,dcate])
    return model