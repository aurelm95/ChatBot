import RedNeuronal
import train
import nlp

all_words,X,Y=train.generate_trainning_data()


Red=RedNeuronal.RedNeuronal([len(all_words),8,8,7])

"""
ignore_words = ['?', '.', '!']
frase="hello, how are you?"
lista=nlp.tokenize(frase)
lista=[nlp.stem(w) for w in lista if w not in ignore_words]
bag=nlp.bag_of_words(lista,all_words)
p=Red.prealimentacion(bag)
"""
# Red.DescensoGradienteEstocastico(zip(X,Y),10000,8,0.001)
# Red.evaluate(zip(X,Y))