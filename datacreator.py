import numpy as np
import csv
from random import random
import unidecode
import string


class DataCreator:
    names_collapsed = [] #without repeat
    amounts_collapsed = []
    
    cities = [] #without repeat
    
    def load_names(self):
        # gender (1 man 2 woman) ; name ; date ; county ; number
        with open("data/dpt2018.csv", encoding="utf8") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            names = [] #with repeat
            
            corresponding_date = []
            corresponding_amount = []
            corresponding_gender = []
            prev = None
            last_row = None
            current_amount = 0
            for row in csvreader: #does not add the last name, isn't a big deal given the size of the dataset
                row = row[0].split(";")
                name = unidecode.unidecode(row[1]).lower()
                
                try:
                    amount = int(row[4])
                    
                    if row[1][0] == '_':continue #excludes names "_PRENOMS_RARES"
                    if (name,int(row[2])) == prev:
                        current_amount += amount
                    else:
                        if last_row != None:
                            names.append(last_row[1])
                            corresponding_date.append(int(last_row[2]))
                            corresponding_amount.append(current_amount)
                            corresponding_gender.append(int(last_row[0])-1)
                        current_amount = amount
                        prev = name,int(row[2])
                        last_row = row
                        last_row[1] = name
                except:
                    continue
        
        print(str(len(names))+' entries')
        prev = None
        current_amount = 0
        for i,n in enumerate(names):
            amount = corresponding_amount[i]
            if n == prev:
                current_amount += amount
            else:
                if prev != None:
                    name = ''.join([i for i in prev if i.isalpha() or i == '-' or i==' '])
                    self.names_collapsed.append(name)
                    self.amounts_collapsed.append(current_amount)
                current_amount = amount
                prev = n
        print(str(len(self.names_collapsed))+' distinct names')
    
    
    def load_cities(self):
        with open("data/villes_france.csv", encoding="utf8") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=";")
            
            for row in csvreader: 
                row = row[0].split(",")
                name = unidecode.unidecode(row[3]).lower()
                name = ''.join([i for i in name if i.isalpha() or i == '-' or i==' '])
                self.cities.append(name)
                
    def save_sample_names(self,size,name):
        np_names = np.array(self.names_collapsed)
        np_amount = np.array(self.amounts_collapsed)
        prob = np_amount/np.sum(np_amount)
        
        sample = np.random.choice(np_names,size=size,p=prob)
        np.save(name,sample)
    
    def save_sample_cities(self,size,name):
        np_cities = np.array(self.cities)
        
        sample = np.random.choice(np_cities,size=size)
        np.save(name,sample)    
    
    def get_vocab(self):
        vocab = ['','>',' ','-']+list(string.ascii_lowercase) #end and start token
        print(str(len(vocab))+ ' letters')
        dic = {}
        for i,c in enumerate(vocab):
            dic[c] = i
        return vocab,dic
    
    def get_one_hot(self,words):
        max_length = np.max(np.vectorize(len)(words))

        vocab,dic = self.get_vocab()
        m = words.shape[0]

        X = np.zeros((m,int(max_length)+10,len(vocab))) #zeros padding at the end

        for i in range(m):
            for j in range(X.shape[1]):
                if j < len(words[i]):
                    c = words[i][j]
                    X[i,j,dic[c]] = 1
                else:
                    X[i,j,0] = 1

        return X