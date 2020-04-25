from tensorflow.keras.layers import Input, Dense,LSTM,Reshape, Lambda, TimeDistributed, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def get_simple_rnn(vocab_size,learning_rate = 0.001):
    X_input = Input(shape=(None,vocab_size))
    
    X = SimpleRNN(vocab_size,activation='softmax',return_sequences=True)(X_input)
    
    model = Model(inputs= X_input, outputs=X)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def get_rnn(vocab_size, na,learning_rate = 0.001):
    X_input = Input(shape=(None,vocab_size))
    
    X = SimpleRNN(na,activation='tanh',return_sequences=True)(X_input)
    X = TimeDistributed(Dense(vocab_size, activation='softmax'))(X)
    
    model = Model(inputs= X_input, outputs=X)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def get_lstm(vocab_size, na,learning_rate = 0.001):
    X_input = Input(shape=(None,vocab_size))
    X = LSTM(na,return_sequences=True)(X_input)
    X = TimeDistributed(Dense(vocab_size, activation='softmax'))(X)
    
    model = Model(inputs= X_input, outputs=X)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model