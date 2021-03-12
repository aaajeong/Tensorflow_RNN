#!/usr/bin/env python
# coding: utf-8

# In[43]:


import tensorflow as tf
import numpy as np

char_arr = [c for c in "SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑봉구우루"]
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)
print ("num_dic:", num_dic)

seq_data = [['word', "단어"], ["wood", "나무"], ["game", "놀이"], ["girl", "소녀"], 
            ["kiss", "키스"], ["love", "사랑"], ["bong", "봉구"], ["uruu", "우루"]]


# In[44]:


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []
    
    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ("S" + seq[1])]
        target = [num_dic[n] for n in (seq[1] + "E")]
        
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)
        
    print("input: ",input)
    print("output: ",output)
    print("target: ",target)

    return input_batch, output_batch, target_batch


# In[45]:


learning_rate = 0.001
n_hidden = 128
total_epoch = 1000

n_class = n_input = dic_len


# In[46]:


enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])


# In[47]:


# encoder: [batch size, time steps, input size]
# decoder: [batch size, time steps]

with tf.variable_scope("encode"):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
    
with tf.variable_scope("decode"):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    
    outputs, dec_stats = tf.nn.dynamic_rnn(dec_cell, dec_input, 
                                           initial_state=enc_states, dtype=tf.float32)


# In[ ]:


model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model, labels=targets
    )
)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[ ]:


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

input_batch, output_batch, target_batch = make_batch(seq_data)

cost_val = []
for epoch in range(total_epoch):
    _, loss = sess.run([opt, cost], feed_dict={enc_input: input_batch,
                                               dec_input: output_batch,
                                               targets: target_batch})
    cost_val.append(loss)
    
    if (epoch+1) % 200 ==0:
        print("Epoch: {:04d}, cost: {}".format(epoch+1, loss))
    
    
print("\noptimization complete")


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(20, 10))
plt.title("cost")
plt.plot(cost_val, linewidth=1, alpha=0.8)
# plt.show()


# In[ ]:




def translate(word):
    seq_data = [word, "P" * len(word)]
    print('seq_data = ',  seq_data)
    
    input_batch, output_batch, target_batch = make_batch([seq_data])

    prediction = tf.argmax(model, 2)
    
    

    result = sess.run(prediction, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})
    
    print('result:',result)
    test = sess.run(model, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})
    print('model test:', test)

    int_model = tf.cast(test, tf.int64)
    int_value_model = sess.run(int_model, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})
    print('int_value_model: ', int_value_model)
    
    decoded = [char_arr[i] for i in result[0]]
    print('decoded:', decoded)
    try:
        end = decoded.index("E")
        translated = "".join(decoded[:end])
        print('end:', end)
        return translated
        
    except Exception as ex:
        pass


# In[ ]:


# print("\n ==== translate test ====")

print("word -> {}".format(translate("word")))
print("wodr -> {}".format(translate("wodr")))
# print("love -> {}".format(translate("love")))
# print("loev -> {}".format(translate("loev")))
# print("bogn -> {}".format(translate("bogn")))
# print("uruu -> {}".format(translate("uruu")))
# print("abcd -> {}".format(translate("abcd")))



# In[ ]:





# In[ ]:





# In[ ]:




