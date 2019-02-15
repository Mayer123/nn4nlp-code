from collections import defaultdict
import time
import random
import torch
import torch.utils.data
import tqdm
import numpy as np
from collections import Counter

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2]

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
PAD = w2i["<PAD>"]
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag], i)

# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
nwords = len(w2i)
ntags = len(t2i)

train = TextDataset(train)
dev = TextDataset(dev)

train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=4, collate_fn=lambda batch : zip(*batch))
dev_loader = torch.utils.data.DataLoader(dev, batch_size=64, num_workers=1, collate_fn=lambda batch : zip(*batch))

# Define the model
EMB_SIZE = 300
FILTER_SIZE = 100
WIN_SIZE = 5

model1 = torch.load("modelatt_seed6")
model2 = torch.load("modelatt_seed2")
model3 = torch.load("modelatt_seed3")
model4 = torch.load("modelatt_seed4")
model5 = torch.load("modelatt_seed5")

use_cuda = torch.cuda.is_available()

if use_cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()
    model5.cuda()

def pad_sequence(sentences):
    max_len = max([len(sent) for sent in sentences])
    if max_len < WIN_SIZE: 
        max_len = WIN_SIZE
    sent_batch = np.zeros((len(sentences), max_len), dtype=int)
    masks = np.zeros((len(sentences), max_len), dtype=int)
    for i, sent in enumerate(sentences):
        sent_batch[i,:len(sent)] = np.array(sent)
        masks[i,:len(sent)] = 1
    return sent_batch, masks

test_correct1 = 0.0
test_correct2 = 0.0
test_correct3 = 0.0
test_correct4 = 0.0
test_correct5 = 0.0
ensemble_correct = 0.0
originals = []
predictions = []
indices = []
for words, tag, idx in dev_loader:
    words, masks = pad_sequence(words)
    #words_tensor = torch.tensor(words).type(type)
    words_tensor = torch.as_tensor(words)
    masks_tensor = torch.as_tensor(masks, dtype=torch.float32)
    tag_tensor = torch.as_tensor(tag)
    if use_cuda:
        words_tensor = words_tensor.cuda()
        masks_tensor = masks_tensor.cuda()
        tag_tensor = tag_tensor.cuda()
    scores1 = model1.evaluate(words_tensor, masks_tensor)
    predict1 = scores1.argmax(dim=1)
    scores2 = model2.evaluate(words_tensor, masks_tensor)
    predict2 = scores2.argmax(dim=1)
    scores3 = model3.evaluate(words_tensor, masks_tensor)
    predict3 = scores3.argmax(dim=1)
    scores4 = model4.evaluate(words_tensor, masks_tensor)
    predict4 = scores4.argmax(dim=1)
    scores5 = model5.evaluate(words_tensor, masks_tensor)
    predict5 = scores5.argmax(dim=1)

    originals += tag
    indices += idx
    ens = []
    for i in range(predict1.size()[0]):
        c = Counter([predict1[i].item(), predict2[i].item(), predict3[i].item(), predict4[i].item(), predict5[i].item()])
        ens.append(c.most_common()[0][0])

    predictions += ens

    ens_predict = torch.as_tensor(ens)
    if use_cuda:
        ens_predict = ens_predict.cuda()

    test_correct1 += torch.sum(torch.eq(predict1, tag_tensor)).item()
    test_correct2 += torch.sum(torch.eq(predict2, tag_tensor)).item()
    test_correct3 += torch.sum(torch.eq(predict3, tag_tensor)).item()
    test_correct4 += torch.sum(torch.eq(predict4, tag_tensor)).item()
    test_correct5 += torch.sum(torch.eq(predict5, tag_tensor)).item()
    ensemble_correct += torch.sum(torch.eq(ens_predict, tag_tensor)).item()

print("Test acc=%.4f" % (test_correct1 / len(dev)))
print("Test acc=%.4f" % (test_correct2 / len(dev)))
print("Test acc=%.4f" % (test_correct3 / len(dev)))
print("Test acc=%.4f" % (test_correct4 / len(dev)))
print("Test acc=%.4f" % (test_correct5 / len(dev)))
print("Test acc=%.4f" % (ensemble_correct / len(dev)))

i2t = {}
for k, v in t2i.items():
    i2t[v] = k

confusion = np.zeros((len(t2i), len(t2i)))

label_count = defaultdict(int)
for i, p in enumerate(predictions):
    if p != originals[i]:
        #label_count[i2t[p]] += 1
        confusion[originals[i]][p] += 1

print (confusion)
for i in range(len(t2i)):
    print (i2t[i])
# with open('test.txt', 'w') as fout:
#     for i, p in enumerate(predictions):
#         fout.write('%s\t%s\t%s\n' % (i2t[p], i2t[originals[i]], indices[i]))



