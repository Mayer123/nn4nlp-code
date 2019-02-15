from collections import defaultdict
import time
import random
import torch
import torch.utils.data
import tqdm
import numpy as np
from models import CNNclass, CNNSimpleAtt


seed = 6
np.random.seed(seed)
torch.manual_seed(seed)

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1]

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
PAD = w2i["<PAD>"]
UNK = w2i["<unk>"]

word_count = defaultdict(int)
with open("topicclass/topicclass_train.txt", "r") as f:
    for line in f:
        tag, words = line.lower().strip().split(" ||| ")
        for w in words.split(" "):
            word_count[w] += 1

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


def generate_embeddings(filename, word_dict):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict), 300))
    count = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split()
            if tokens[0] in word_dict:
                embeddings[word_dict[tokens[0]]] = np.array(list(map(float, tokens[1:])))
                count += 1
    print (count, len(word_dict))
    return embeddings


# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
nwords = len(w2i)
ntags = len(t2i)

print (nwords)
print (ntags)
print (train[0])

train = TextDataset(train)
dev = TextDataset(dev)
print (len(train))
train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=4, collate_fn=lambda batch : zip(*batch))
dev_loader = torch.utils.data.DataLoader(dev, batch_size=64, num_workers=4, collate_fn=lambda batch : zip(*batch))

# Define the model
EMB_SIZE = 300
FILTER_SIZE = 100
WIN_SIZE = 5

embeddings = generate_embeddings('crawl-300d-2M.vec', w2i)

# initialize the model
model = CNNclass(nwords, embeddings, EMB_SIZE, FILTER_SIZE, ntags)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    torch.cuda.manual_seed(seed)
    model.cuda()

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

global_step = 0
best = 0.0
for ITER in range(3):
    # Perform training
    #random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for words, tag in tqdm.tqdm(train_loader):
        global_step += 1
        
        words, masks = pad_sequence(words)
        words_tensor = torch.as_tensor(words)
        masks_tensor = torch.as_tensor(masks, dtype=torch.float32)
        tag_tensor = torch.as_tensor(tag)
        if use_cuda:
            words_tensor = words_tensor.cuda()
            tag_tensor = tag_tensor.cuda()
            masks_tensor = masks_tensor.cuda()
        scores = model(words_tensor, masks_tensor)
        predict = scores.argmax(dim=1)
        
        train_correct += torch.sum(torch.eq(predict, tag_tensor)).item()
        my_loss = criterion(scores, tag_tensor)
        train_loss += my_loss.item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()

        if global_step % 1000 == 0:
            # Perform testing
            test_correct = 0.0
            for words, tag in dev_loader:
                words, masks = pad_sequence(words)
               
                words_tensor = torch.as_tensor(words)
                masks_tensor = torch.as_tensor(masks, dtype=torch.float32)
                tag_tensor = torch.as_tensor(tag)
                if use_cuda:
                    words_tensor = words_tensor.cuda()
                    masks_tensor = masks_tensor.cuda()
                    tag_tensor = tag_tensor.cuda()
                scores = model.evaluate(words_tensor, masks_tensor)
                predict = scores.argmax(dim=1)
                test_correct += torch.sum(torch.eq(predict, tag_tensor)).item()
            print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))
            if test_correct > best:
                best = test_correct
                torch.save(model, 'model_seed%s' % str(seed))
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss / len(train), train_correct / len(train), time.time() - start))
    