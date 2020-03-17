# Load the data

import csv
import random


#coding for the labels: 0 = only climate change, 1 = both, 2 = only health
# Map this to two labels for now: 0 = either only climate OR only health, 1 = both
easy_label_map = {0:0, 1:1, 2:0}

def load_sst_data(filename):
  reader = csv.reader(open(filename, 'r'), delimiter='\t')
  data = []
  for row in reader:
    example = {}
    k, v = row
    example['label'] = easy_label_map[int(v)]
    if example['label'] is None:
      continue
    text = re.sub(r'\s*(\(\d)|(\))\s*', '', k)
    example['text'] = text
    data.append(example)
      
    random.seed(1)
    random.shuffle(data)
  return data

sst_home = 'lancet.csv'  
data_set = load_sst_data(sst_home)

[example["text"] for example in data_set if example["label"] == 0][:10]

[example["text"] for example in data_set if example["label"] == 1][:10]

