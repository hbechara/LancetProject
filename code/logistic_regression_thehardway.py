
# Load the data

import re
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
#tensorflow_version 2.x
import tensorflow as tf
tf.__version__

# posititve/negative classification
easy_label_map = {0:0, 1:0, 2:None, 3:1, 4:1}

def load_sst_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            example['label'] = easy_label_map[int(line[1])]
            if example['label'] is None:
                continue
            
            # Strip out the parse information and the phrase labels---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data


#bag-of-words feature vectors
def feature_function(datasets):
    '''Annotates datasets with feature vectors.'''
    
    # Extract vocabulary
    def tokenize(string):
        return string.split()
    
    word_counter = collections.Counter()
    for example in datasets[0]:
        word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter])
                                
    feature_names = set()
    for i, dataset in enumerate(datasets):
        for example in dataset:
            example['features'] = collections.defaultdict(float)
            
            # Extract features
            word_counter = collections.Counter(tokenize(example['text']))
            for x in word_counter.items():
                if x[0] in vocabulary:
                    example["features"]["word_count_for_" + x[0]] = x[1]
            
            feature_names.update(example['features'].keys())
                            
    # assign indices
    feature_indices = dict(zip(feature_names, range(len(feature_names))))
    indices_to_features = {v: k for k, v in feature_indices.items()}
    dim = len(feature_indices)
                
    # create vectors
    for dataset in datasets:
        for example in dataset:
            example['vector'] = np.zeros((dim))
            for feature in example['features']:
                example['vector'][feature_indices[feature]] = example['features'][feature]
    return indices_to_features, dim

def evaluate_classifier(classifier, eval_set):
    correct = 0
    hypotheses = classifier(eval_set)
    for i, example in enumerate(eval_set):
        hypothesis = hypotheses[i]
        if hypothesis == example['label']:
            correct += 1        
    return correct / float(len(eval_set))

# Define Logistic Regression model
class logistic_regression_classifier:
    def __init__(self, dim, lr=1.0, reg_w=0.0, num_ep=50 ):
        # Define the hyperparameters
        self.learning_rate = lr  
        self.reg_weight = reg_w # Regularization weight 
        self.training_epochs = num_ep  # How long to train for
        self.display_epoch_freq = 1  # How often to test and print out statistics
        self.dim = dim  # The number of features
        self.batch_size = 256  
        
        self.trainable_variables = []
        # Define the model
        self.W = tf.Variable(tf.zeros([self.dim, 2]))   # if two class classification
        self.b = tf.Variable(tf.zeros([2]))
        self.trainable_variables.append(self.W)
        self.trainable_variables.append(self.b)
    def model(self,x):
        logits = tf.matmul(x, self.W) + self.b
        return logits
        
    def train(self, training_data, dev_set):
        def get_minibatch(dataset, start_index, end_index):
            indices = range(start_index, end_index)
            vectors = np.float32(np.vstack([dataset[i]['vector'] for i in indices]))
            labels = [dataset[i]['label'] for i in indices]
            return vectors, labels

        print('Training.')

        # Training cycle
        stats = dict()
        stats['loss']=[]
        stats['dev']=[]
        stats['train']=[]

        for epoch in range(self.training_epochs):
            random.shuffle(training_set)
            avg_cost = 0.
            total_batch = int(len(training_set) / self.batch_size)
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_vectors, minibatch_labels = get_minibatch(training_set, 
                                                                    self.batch_size * i, 
                                                                    self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                with tf.GradientTape() as tape:
                  logits = self.model(minibatch_vectors)
                  regularizer = tf.nn.l2_loss(self.W)
                  cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=minibatch_labels)+self.reg_weight*regularizer)
                # This performs the SGD update equation
                gradients = tape.gradient(cost, self.trainable_variables)
                optimizer = tf.optimizers.SGD(self.learning_rate)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                # Compute average loss
                avg_cost += cost / (total_batch * self.batch_size)
              

                
            # Display some statistics about the step
            if (epoch+1) % self.display_epoch_freq == 0:
              stats['loss'].append(avg_cost)
              stats['dev'].append(evaluate_classifier(self.classify, dev_set[0:500]))
              stats['train'].append(evaluate_classifier(self.classify, training_set[0:500]))
              #  tf.print("Epoch:", (epoch+1), "Cost:", avg_cost,
               #       "Dev acc:", evaluate_classifier(self.classify, dev_set[0:500]), 
                #      "Train acc:", evaluate_classifier(self.classify, training_set[0:500]))
        return stats
    
    def classify(self, examples):
        # This classifies a list of examples
        vectors = np.float32(np.vstack([example['vector'] for example in examples]))
        logits = self.model(vectors)
        return np.argmax(logits, axis=1)

def plot(stats):
  fig,ax = plt.subplots(2,1,sharex=True)
  loss_train = stats['loss']
  acc_eval = stats['dev']
  acc_train = stats['train']
  epochs = range(len(stats['loss']))
  ax[1].plot(epochs, loss_train, 'g', label='Training loss')
  ax[0].plot(epochs, acc_eval, 'b', label='validation accuracy')
  ax[0].plot(epochs, acc_train, 'r', label='training accuracy')
  ax[0].set_title('Training and Evaluation accuracy')
  ax[1].set_title('Training loss')
  plt.xlabel('Epochs')
  ax[1].set_ylabel('Loss')
  ax[0].set_ylabel('Accuracy')
  ax[0].legend()
  ax[1].legend()
  plt.show()



sst_home = 'drive/My Drive/Colab Notebooks/2019-2020_labs/data/trees/'  
training_set = load_sst_data(sst_home + '/train.txt')
dev_set = load_sst_data(sst_home + '/dev.txt')
test_set = load_sst_data(sst_home + '/test.txt')

indices_to_features, dim = feature_function([training_set, dev_set, test_set])

classifier = logistic_regression_classifier(dim,0.1,0.1,50)
stats = classifier.train(training_set, dev_set)

for i in [0,0.01,0.1]:
  classifier = logistic_regression_classifier(dim,0.1,i,100)
  stats = classifier.train(training_set, dev_set)
  plot(stats)

#apparently 0.1 is a good reguliser weight as the model does not overtrain
classifier = logistic_regression_classifier(dim,0.1,0.01,100)
stats = classifier.train(training_set, dev_set)
plot(stats)


evaluate_classifier(classifier.classify, test_set)

