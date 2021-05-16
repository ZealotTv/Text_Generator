import numpy 
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open('C:\\Users\\ZealotTV\\Downloads\\работа\\не соц сеть\\esdasd\\mml-book.txt').read()

def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs = tokenize_words(file)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

seq_length = 100
x_data = []
y_data = []

input_len = len(processed_inputs)
vocab_len = len(chars)

for i in range(0, input_len - seq_length, 1):

    in_seq = processed_inputs[i:i + seq_length]

    out_seq = processed_inputs[i + seq_length]


    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))



filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]


model.fit(X, y, epochs=20, batch_size=256, callbacks=desired_callbacks)

"""
В начале было 4 эпохи. Результат:
al understand ing general change variables approach theorem 6 16 change variables probability relies 
ependent number features stead number parameters increases number examples training set saw similar 
tor x � r784 examples digits shown figure 10 3 figure 10 12 effect increasing number principal origi
78 � 0 48 � 0 62 0 62 0 0 start set vectors x colored dots see top left panel fig ure 4 9 arranged g  
"""
# filename = "model_weights_saved.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# num_to_char = dict((i, c) for i, c in enumerate(chars))

# start = numpy.random.randint(0, len(x_data) - 1)
# pattern = x_data[start]
# print("Random Seed:")
# print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
