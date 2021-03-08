from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 문자열이 너무 길어서 다음과 같이 잘라서 학습 (윈도우를 옮겨가면서 정해진 시퀀스만큼 자른다.)
# 0 if you wan -> f you want
# 1 f you want -> you want
# 2 you want -> you want t
# 3 you want t -> ou want to
# ...
# 168 of the se -> of the sea
# 169 of the sea -> f the sea

char_set = list(set(sentence)) #문장에서 유일한 문자들을 리스트로.
char_dic = {w: i for i, w in enumerate(char_set)}   #딕셔너리 형태 {'f':0, 'm':1,...}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

dataX = []
dataY = []

# 시퀀스를 움직이면서 x, y data 을 뽑아낸다.
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    # 문자를 char_dic 에 있는 인덱스로.
    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    # 최종 data X,Y 을 저장.
    dataX.append(x)
    dataY.append(y) 

batch_size = len(dataX) # 데이터가 많을 경우, batch size 을 넣어준다. (이 경우 한번에 다 넣어줌.)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape


# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

# 긴 문장의 경우 여러개의(깊게) rnn 층을 쓰도록 해보자
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer(softmax)
# RNN 에서 나온 값을 그대로 사용하지 않고, softmax 을 사용한 output 을 사용해보자.
# -> rnn 에서 나온 output 을 softmax 에 넣을 수 있도록 쌓는다.(reshape 한다.)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])

# softmax
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
# softmax 의 output 을 다시 펼친다.(reshape 한다.), rnn 의 output 과 같은 shpae
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

# loss
# 이제 logits 에는 softmax 가 거친 output 이 들어가게 된다. (이 점을 주의하면, 좋은 성능을 가질 수 있다.)
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training and print results
for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''
