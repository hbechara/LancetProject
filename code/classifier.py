from pathlib import Path
import pandas as pd 
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset 
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm, tqdm_notebook #only for the progress bars
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def read_dataset(path):
    #coding for the labels: 0 = only climate change, 1 = both, 2 = only health
    # Map this to two labels for now: 0 = either only climate OR only health, 1 = both
    label_mapping = {0:0, 1:1, 2:0}
    label_map = lambda x: label_mapping[int(x)]
    joint = []
    for filename in path:
        frame = pd.read_csv(filename, index_col=None, header=0, sep='\t', converters={'label':label_map},error_bad_lines = False, names=['paragraph','label'])
        joint.append(frame)
    df = pd.concat(joint, axis=0, ignore_index=True)
    return df


class Sequences(Dataset):
    def __init__(self, path):               
        df = read_dataset(path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(df.paragraph.tolist())
        self.labels = df.label.tolist()
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
       # print(self.token2idx)
       # print(self.idx2token)

    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]
    
    def __len__(self):
        return self.sequences.shape[0]



class Classifier(nn.Module):
    def __init__(self, vocab_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=vocab_size, 
                             out_features=1)

    
    def forward(self, inputs,  apply_sigmoid=False):
        y_out = self.fc1(inputs).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
    
def predict(text):
    model.eval()
    with torch.no_grad(): # setting all grad flags to false
        test_vector = torch.LongTensor(dataset.vectorizer.transform([text]).toarray())

        output = model(test_vector.float())
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: BOTH climate change and health')
        else:
            print(f'{prediction:0.3}: EITHER climate change OR health')

#%%

#Load the csv files into one dataset
DATA_PATH = r'../csv'
filenames = glob.glob(DATA_PATH + "/*.csv")


dataset_frame = read_dataset(filenames)
print(dataset_frame.sample(5))

#%%
dataset = Sequences(filenames)
train_loader = DataLoader(dataset, batch_size=4096)


#%%
#initialise the model
model = Classifier(len(dataset.token2idx))

#Define optimizer and Logits
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

#train the model
model.train()
train_losses = []
for epoch in range(100):
    progress_bar = tqdm_notebook(train_loader, leave=False) #
    losses = [] #
    total = 0
    for inputs, target in progress_bar:
        model.zero_grad()
        output = model(inputs.float())
        loss = criterion(output.squeeze(), target.float())        
        loss.backward()              
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()        
        progress_bar.set_description(f'Loss: {loss.item():.3f}')        
        losses.append(loss.item())
        total += 1    
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)        
    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')
   # print("training pass", epoch, "loss", epoch_loss)


test_text = """
CLimate change is a thread to all of us.
"""
predict(test_text)

test_text = """
An increasing amount of emissions will have negative implications for lung diseases such as pneunomia.
"""
predict(test_text)