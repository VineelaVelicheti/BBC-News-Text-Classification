
# coding: utf-8

# In[28]:


import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.contrib import learn
import re
import csv


# In[29]:


def clean_str(string):
   
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"ï»¿", "", string)
   
    return string.strip().lower()


# In[30]:


def load_data_and_labels(data_folder):
    
    folders = ["business","entertainment","politics","sport","tech"]

    os.chdir(data_folder)
    
    x = []
    y = []

    for i in folders:
        files = os.listdir(i)
        for text_file in files:
            file_path = i + "/" +text_file
            
            with open(file_path, "r") as f:
                data = f.readlines()
            data = ' '.join(data)
            x.append(data)
            y.append(i)
    
    business_label = []
    entertainment_label = []
    politics_label = []
    sport_label = []
    tech_label = []
    
    
    x = [clean_str(sent) for sent in x]
    
    for index, entry in enumerate(y):
        if(y[index] == "business"): 
            business_label.append([1,0,0,0,0])
        elif(y[index] =="entertainment"):
            entertainment_label.append([0,1,0,0,0])
        elif(y[index] =="politics"):
            politics_label.append([0,0,1,0,0])
        elif(y[index] =="sport"):
            sport_label.append([0,0,0,1,0])
        elif(y[index] =="tech"):
            tech_label.append([0,0,0,0,1])
        
    y = np.concatenate([business_label, entertainment_label,politics_label,sport_label,tech_label], 0)
    return[x,y]


# In[31]:


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            #np.arrange(5)=[0,1,2,3,4]
            #np.random.permutation randomizes the order
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
        


# In[32]:


#importing test data
x_raw, y_test = load_data_and_labels("/Users/vineevineela/Desktop/bbc/test")
y_test = np.argmax(y_test, axis=1)
    

# Map data into vocabulary
vocab_path = os.path.join("/Users/vineevineela/Desktop/bbc/runs/1526337975","vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


# In[33]:


print("\nTesting...\n")

checkpoint_file = tf.train.latest_checkpoint("/Users/vineevineela/Desktop/bbc/runs/1526337975/checkpoints")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), 10, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

correct_predictions = float(sum(all_predictions == y_test))

print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join("/Users/vineevineela/Desktop/bbc","prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

