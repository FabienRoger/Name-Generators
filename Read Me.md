I compare the effectiveness of different types of RNN and word generation on the well-known task of genaratgin new names.

I generate new french first names using the a dataset containing over 15.000 distinct french names (of which I select a 100.000 thousands names in the right proportion) as well as french city names using a dataset containg all 36.000 city names (of which I randomly sampled 30.000).

The csv files used can be found at :
https://www.insee.fr/fr/statistiques/2540004#consulter
and
https://sql.sh/736-base-donnees-villes-francaises

(The inputs are one-hot vectors representing all letters from a to z as well as space, and -)

The RNN compared are :
- A simple RNN with a softmax output (rnn1) (1.830 trainable parameters)
- A RNN with 50 hidden units (activated by tanh) and a distributed dense softmax output (rnn2) (5.580 trainable parameters)
- A RNN with 100 hidden units (activated by tanh) and a distributed dense softmax output (rnn3) (16.130 trainable parameters)
- A LSTM with 50 hidden units and a distributed dense softmax output (17.730 trainable parameters)

The two word generating approaches are :
- The naïve "pick the next letter according to the probability distribution of the softmax output" (might return names in the training dataset)
- An approach in which I pick the next letter according to the weight p^3 (where p is the probability the softmax layer outputs). It makes it more likely that the most likely letter is picked. I only return completely new words (not in the training dataset), and also a number, the ratio r = number of new words generated / words generated.

Each network was trained for approximately 50 epochs.

Results for first names generation :
Random sample from the dataset : 
roger, marie, victor, jocelyne, charlotte, laetitia, yassir, anthony, edgar, fernand

First approach results, rnn :
etrenne, chrone, scelle, uis, riste, vor, niel, inel, soch, xyve
First approach results, rnn2 :
ella, lesaule, iserie, rem, esmo, zettise, iso, elle, ylena, elline
First approach results, rnn3 :
ol, uil, angie, etie, ance, issohe, isstie, delene, elhind, annire
First approach results, lstm :
richan, andre, olivier, elio, ariano, arnandine, andon, henri, orien, olvine

Second approach results, rnn :
lien, anie, uis, adie, ande, ystene, ristie, vilie, zele, andie (r=0.4)
Second approach results, rnn2 :
elline, delbrine, ise, alba, deride, iso, ista, delbrin, delben, alins (r=0.42)
Second approach results, rnn3 :
ine, ule, inetie, inettet, ya, ole, issain, yas, inette, uil (r=0.14)
Second approach results, lstm :
orienne, udy, arnand, odinette, odine, orien, regne, olivie, ande, orene (r=0.06)

Comments :
To a french speaker, the names produced by rnn, rnn2, and rnn3 are not very good. The one produced by lstm are original and pretty convincing, especially with the second approach. r is very low for rnn3 and lstm, meaning that the neural network produced mostly names already existing. This might be due to overfitting, but also to the fact that the number of names looking like names in french is quite small, and that therefore the network just learned what a french name looks like.

Results for city names generation :
Random sample from the dataset : 
eaubonne, merlimont, cheroy, nernier, outremecourt, sainte-marie-aux-mines, laval-du-tarn, bray-sur-somme, pact, bresson

First approach results, rnn :
vanq, uds-hegret-aury, viru, nies-gncemex, , oury, , , nere, wien-er-rain
First approach results, rnn2 :
illes, lone, azeuil-en-ville-sur-beoux, ayses-palliers, auvilles, usxiain, ouzieres, usclers-sous-pors, urisg, ezles
First approach results, rnn3 :
reilly, azonnediere, eux-loubard, rectensenomarde, auance, iry-camille-dampeaux, rennay, uvantre, gnanches-debis, risind
First approach results, lstm :
doualde, menheilles, piernes, ullon, vigniec, udo, zelly, yelon, tailles, indoux

Second approach results, rnn :
andes, ren, aint-er-mancy, hille, ran, aint-dr-de-lache, aint, are, an (r=0.17)
Second approach results, rnn2 :
urieres, uppe, apille-sur-cotel, urberes, ures, apent-de-berieres, empres, aberrieres, ange, apelle-sur-seine (r=1)
Second approach results, rnn3 :
resseux, rentac, aulines-sur-seille, rembardigne, rembricourt, remines, erville, ressin-sur-corce, rembernes, resnes (r=1)
Second approach results, lstm :
reilly-les-bains, arcelles-sur-orge, ressey, laisee-sur-mer, eschaux, anchaux, esse, isse-les-bonges, usses-sur-loire, holing (r=0.77)

Comments :
To a french speaker, the city names produced by the second approach are much better than the names produced by the second one. City names produced by rnn3 and lstm might even seem more real than the random sample (probably because they are closer to the "average city name", which means our network achieved to learn what a french city name looks like, awesome !).

main.py contains the code calling the functions defined in datacreator.py (which loads and creates the dataset, a bit of data preprocessing), models.py (which creates and compiles the models described above) and namegen.py (which generates "words"/names given a trained RNN)






