import json
import numpy as np

from nlp import bag_of_words, tokenize, stem



def generate_trainning_data():
	f=open('intents.json', 'r')
	intents = json.load(f)
	all_words = []
	classes = []
	trainning_data = []
	for intent in intents['intents']:
			tag = intent['tag']
			# add to tag list
			classes.append(tag)
			for pattern in intent['patterns']:
					# tokenize each word in the sentence
					w = tokenize(pattern)
					# add to our words list
					all_words.extend(w)
					# add to xy pair
					trainning_data.append((w, tag))

	# stem and lower each word
	ignore_words = ['?', '.', '!']
	all_words = [stem(w) for w in all_words if w not in ignore_words]
	# remove duplicates and sort
	all_words = sorted(set(all_words))
	classes = sorted(set(classes))
	"""
	print(len(trainning_data), "patterns")
	print(len(classes), "classes:", classes)
	print(len(all_words), "unique stemmed words:", all_words)
	"""
	# create training data
	X_train = []
	y_train = []
	for (pattern_sentence, tag) in trainning_data:
			# X: bag of words for each pattern_sentence
			bag = bag_of_words(pattern_sentence, all_words)
			bag=bag.reshape(len(bag),1)
			X_train.append(bag)
			# y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
			label = classes.index(tag)
			y_train.append(label)

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	return all_words,X_train,y_train

if __name__=='__main__':
	generate_trainning_data()