# coding: utf-8
from datacreator import *
from models import *
from namegen import *

dc = DataCreator()
vocab,dic = dc.get_vocab()
vocab_size = len(vocab)

rnn = get_simple_rnn(vocab_size,learning_rate=0.01)
rnn2 = get_rnn(vocab_size,50,learning_rate=0.01)
rnn3 = get_rnn(vocab_size,100,learning_rate=0.01)
lstm = get_lstm(vocab_size,50,learning_rate=0.01)
print(rnn.summary())
print(rnn2.summary())
print(rnn3.summary())
print(lstm.summary())


dc.load_names()
dc.save_sample_names(100000,'data/100k names')

sample = np.load('data/100k names.npy')
print(','.join(list(sample[:10])))
X = dc.get_one_hot(sample)

Y = np.zeros(X.shape)
Y[:,:-1,:] = X[:,1:,:]
Y[:,-1,0] = 1

#rnn.load_weights('models/weights 1.nn')
rnn.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnn.save_weights('models/weights 1.nn')


#rnn2.load_weights('models/weights 2.nn')
rnn2.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnn2.save_weights('models/weights 2.nn')

#rnn3.load_weights('models/weights 3.nn')
rnn3.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnn3.save_weights('models/weights 3.nn')

#lstm.load_weights('models/weights lstm.nn')
lstm.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
lstm.save_weights('models/weights lstm.nn')

rnn.load_weights('models/weights 1.nn')
rnn2.load_weights('models/weights 2.nn')
rnn3.load_weights('models/weights 3.nn')
lstm.load_weights('models/weights lstm.nn')

print(', '.join(list(sample[:10])))

s = generate_words(rnn,10,vocab,dic)
s2 = generate_words(rnn2,10,vocab,dic)
s3 = generate_words(rnn3,10,vocab,dic)
sm = generate_words(lstm,10,vocab,dic)
print(', '.join(s))
print(', '.join(s2))
print(', '.join(s3))
print(', '.join(sm))

''' I get, by taking a random letter from the softmax distribution :
athere,ile,hran,ane,obiro,ugel,,wfilise,edril,sclad
dona,elley,wena,ellene,udo,wellite,alan,bertran,isoe,iso
deandetelliee,ule,rene,wathe,yas,delene,ule,ierne,enz,ole
arie,emilie,idrinr,enwa,udorin,,elisabet,domonn,eline,ugutte

for reference, here is a random sample from the dataset
henri,bruno,muriel,tiana,christine,henriette,alain,jade,patricia,sylvie
'''
sa,r = generate_words_advanced(rnn,10,vocab,dic,sample,3)
sa2,r2 = generate_words_advanced(rnn2,10,vocab,dic,sample,3)
sa3,r3 = generate_words_advanced(rnn3,10,vocab,dic,sample,3)
sam,rm = generate_words_advanced(lstm,10,vocab,dic,sample,3)

#only new names are printed. r is the ratio #new words generated/#words generated during the creation of the list 
print('')
print('{:.2f} : '.format(r)+', '.join(sa))
print('{:.2f} : '.format(r2)+', '.join(sa2))
print('{:.2f} : '.format(r3)+', '.join(sa3))
print('{:.2f} : '.format(rm)+', '.join(sam))


dc.load_cities()
dc.save_sample_cities(30000,'data/30k cities')

rnnc = get_simple_rnn(vocab_size,learning_rate=0.01)
rnn2c = get_rnn(vocab_size,50,learning_rate=0.01)
rnn3c = get_rnn(vocab_size,100,learning_rate=0.01)
lstmc = get_lstm(vocab_size,50,learning_rate=0.01)

sample = np.load('data/30k cities.npy')
X = dc.get_one_hot(sample)

Y = np.zeros(X.shape)
Y[:,:-1,:] = X[:,1:,:]
Y[:,-1,0] = 1

#rnnc.load_weights('models/weights 1c.nn')
rnnc.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnnc.save_weights('models/weights 1c.nn')


#rnn2c.load_weights('models/weights 2c.nn')
rnn2c.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnn2c.save_weights('models/weights 2c.nn')

#rnn3c.load_weights('models/weights 3c.nn')
rnn3c.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
rnn3c.save_weights('models/weights 3c.nn')

#lstmc.load_weights('models/weights lstmc.nn')
lstmc.fit(x = X, y = Y, epochs = 10, batch_size = 128, verbose=2)
lstmc.save_weights('models/weights lstmc.nn')


#rnnc.load_weights('models/weights 1c.nn')
rnn2c.load_weights('models/weights 2c.nn')
rnn3c.load_weights('models/weights 3c.nn')
lstmc.load_weights('models/weights lstmc.nn')

s = generate_words(rnnc,10,vocab,dic)
s2 = generate_words(rnn2c,10,vocab,dic)
s3 = generate_words(rnn3c,10,vocab,dic)
sm = generate_words(lstmc,10,vocab,dic)


print(', '.join(list(sample[:10])))
print('')
print(', '.join(s))
print(', '.join(s2))
print(', '.join(s3))
print(', '.join(sm))

sa,r = generate_words_advanced(rnnc,10,vocab,dic,sample,3)
sa2,r2 = generate_words_advanced(rnn2c,10,vocab,dic,sample,3)
sa3,r3 = generate_words_advanced(rnn3c,10,vocab,dic,sample,3)
sam,rm = generate_words_advanced(lstmc,10,vocab,dic,sample,3)

#only new city names are printed. r is the ratio #new words generated/#words generated during the creation of the list 
print('')
print('{:.2f} : '.format(r)+', '.join(sa))
print('{:.2f} : '.format(r2)+', '.join(sa2))
print('{:.2f} : '.format(r3)+', '.join(sa3))
print('{:.2f} : '.format(rm)+', '.join(sam))
