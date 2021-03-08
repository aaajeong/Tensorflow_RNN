# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility
 
sample = " if you want you"
idx2char = list(set(sample))  # index -> char , set 을 사용해서 각각의 character 을 가져옴 (유일한 문자로.)
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> index (각각의 character 가 숫자로 나오게 됨.)

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

#input, output
sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell 즉, (마지막 -1) 까지
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# one-hot-encoding 
x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])    # weight 은 모두 1로.
#loss 함수
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
#Training
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 출력에 대해 prediction
prediction = tf.argmax(outputs, axis=2)

#Training and Result
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):  # 문장이 길어서 3000번 정도..?
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))


'''
0 loss: 2.35377 Prediction: uuuuuuuuuuuuuuu
1 loss: 2.21383 Prediction: yy you y    you
2 loss: 2.04317 Prediction: yy yoo       ou
3 loss: 1.85869 Prediction: yy  ou      uou
4 loss: 1.65096 Prediction: yy you  a   you
5 loss: 1.40243 Prediction: yy you yan  you
6 loss: 1.12986 Prediction: yy you wann you
7 loss: 0.907699 Prediction: yy you want you
8 loss: 0.687401 Prediction: yf you want you
9 loss: 0.508868 Prediction: yf you want you
10 loss: 0.379423 Prediction: yf you want you
11 loss: 0.282956 Prediction: if you want you
12 loss: 0.208561 Prediction: if you want you

...

'''
