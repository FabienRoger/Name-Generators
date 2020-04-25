import numpy as np

def generate_words(model,number,vocab,dic):
    r = [generate_word(model,vocab,dic) for _ in range(number)]
    return r
    
def generate_word(model,vocab,dic):
    word = np.zeros((1,1,len(vocab)))
    word[0,0,1] = 1
    i = 0
    letter = None
    s = ''
    while letter != 0 and i<30:
        p = model.predict_on_batch(word).numpy()
        
        letter = np.random.choice(len(vocab),p=p[0,i,:])
        
        s += vocab[letter]
        one_hot = np.zeros((1,1,len(vocab)))
        one_hot[0,0,letter] = 1
        word = np.concatenate([word,one_hot],axis = 1)
        
        i += 1
    return s

def generate_words_advanced(model,number,vocab,dic,training_words,first_strength):
    r = []
    
    i = 0
    while len(r)<number and i<100*number:
        w = generate_word_advanced(model,vocab,dic,first_strength)
        if w not in training_words and w not in r:
            r.append(w)
        
        i += 1
    
    return r, number/i

def generate_word_advanced(model,vocab,dic,first_strength): 
    #first_strength corresponds to how much the firsts letters are chosen.
    #1 has the the effect as basic word generator, with infinity (or a large value), the best letter is always chosen
    word = np.zeros((1,1,len(vocab)))
    word[0,0,1] = 1
    i = 0
    letter = None
    s = ''
    while letter != 0 and i<30:
        p = model.predict_on_batch(word).numpy()
        
        strengths = p[0,i,:] ** first_strength;
        strengths = strengths/np.sum(strengths)
        
        letter = np.random.choice(len(vocab),p=strengths)
        
        s += vocab[letter]
        one_hot = np.zeros((1,1,len(vocab)))
        one_hot[0,0,letter] = 1
        word = np.concatenate([word,one_hot],axis = 1)
        
        i += 1
    return s
    