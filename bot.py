import random
import json
import numpy as np

from nlp import bag_of_words, tokenize
import RedNeuronal
import train

all_words,X,Y=train.generate_trainning_data()


Red=RedNeuronal.RedNeuronal([len(all_words),8,8,7])


json_data=open('intents.json', 'r')
intents = json.load(json_data)
classes = []
for intent in intents['intents']:
	classes.append(intent['tag'])


bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(len(X),1)
    Y = Red.prealimentacion(X)
    print(Y)
    Y = np.argmax(Y)
    print("Clase:",classes[Y])
    for intent in intents['intents']:
        if classes[Y] == intent["tag"]:
            print(f"{bot_name}: {random.choice(intent['responses'])}")