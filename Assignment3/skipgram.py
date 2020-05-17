#!/usr/bin/env python
# coding: utf-8

# In[1]:

import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from datetime import datetime

load_from_checkpoint = True
full_load = True

# In[2]:


# corpus = ' '.join([
#     'he is a king',
#     'she is a queen',
#     'he is a man',
#     'she is a woman',
#     'warsaw is poland capital',
#     'berlin is germany capital',
#     'paris is france capital',
# ])
# print(corpus)

def get_corpus_details(file_name="../abc_corpus.txt", window_size=2, min_freq=1):
#     corpus = ' '.join([
#         'he is a king',
#         'she is a queen',
#         'he is a man',
#         'she is a woman',
#         'warsaw is poland capital',
#         'berlin is germany capital',
#         'paris is france capital',
#     ])
    corpus = open(file_name, encoding="utf8").read()
    corpus = ''.join([i if i not in string.punctuation else ' ' for i in corpus.lower()])
    stop_words = set(stopwords.words('english'))
    def tokenize_corpus(corpus):
        tokens = [x for x in corpus.split() if x not in stop_words]
        return tokens

    tokenized_corpus = tokenize_corpus(corpus)
    print("len(tokenized_corpus)", len(tokenized_corpus))

    vocabulary = {}
    for word in tokenized_corpus:
        if word not in vocabulary:
            vocabulary[word] = 0
        vocabulary[word] += 1

#     word2idx = {word: idx for idx, (word, freq) in enumerate(vocabulary.items()) if freq >= min_freq}
    counter = 0
    word2idx = {}
    for word, freq in vocabulary.items():
        if word not in word2idx and freq >=min_freq:
            word2idx[word] = counter
            counter += 1
    idx2word = {idx: word for word, idx in word2idx.items()}

    print(len(word2idx), len(idx2word))

    vocabulary_size = len(word2idx)

    idx_pairs = []
    indices = [word2idx[word] for word in tokenized_corpus if word in word2idx]
    print("max(indices)", max(indices), "len(word2idx)", len(word2idx), "max(word2idx.values())", max(word2idx.values()))
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if center_word_pos != context_word_pos and (context_word_pos >= 0 and context_word_pos < len(indices)):
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

    print("idx_pairs.shape", idx_pairs.shape)
    return idx_pairs, (word2idx, idx2word, vocabulary_size)


def plot_tsne(plot_save_file_dir, label_vectors, labels, epoch="checkpoint"):
    
    print(len(label_vectors), len(labels))
    
    print("\nComputing TSNE...")
    tsne = TSNE(n_components=2, random_state=21, init="pca", verbose=1)
    Y = tsne.fit_transform(label_vectors)
    print("Computed TSNE")
    
    np.save("tsne.npy", Y)
    
#     plt.figure()

#     x_coords = Y[:, 0]
#     y_coords = Y[:, 1]
#     # display scatter plot
#     plt.scatter(x_coords, y_coords)

#     plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
#     plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    
#     plot_save_file_path = "{}/{}.png".format(plot_save_file_dir, epoch)
#     plt.savefig(plot_save_file_path)
#     print("Plot saved at {}".format(plot_save_file_path))
    
    plt.figure()

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

#     for label, x, y in zip(labels, x_coords, y_coords):
#         plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#     plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
#     plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    
    plot_save_file_path = "{}/{}_annotated.png".format(plot_save_file_dir, epoch)
    plt.savefig(plot_save_file_path)
    print("Plot saved at {}".format(plot_save_file_path))


# In[3]:


class PairDataset(Dataset):
    def __init__(self, idx_pairs, word2idx, idx2word):
        self.idx_pairs = idx_pairs
        self.word2idx = word2idx
        self.idx2word = idx2word
        
    def __len__(self):
        return len(self.idx_pairs)
    
    def __getitem__(self, index):
        return self.idx_pairs[index]

class SkipGram(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.encoder = nn.Linear(vocab_size, hidden_size, bias = True)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias = True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def embedding(self, vec):
        vec = vec.view((-1,))
        x = torch.zeros((vec.shape[0], self.vocab_size), device=self.device).float()
        for i in range(vec.shape[0]):
            x[i][vec[i]] = 1
        return x
        
    def forward(self, center_vec, encoding=False):
        center_vector_batch = center_vec.view((1, -1))
        emb = self.embedding(center_vector_batch).view((-1, self.vocab_size))
        hidden_layer = self.encoder(emb)
        res = self.decoder(hidden_layer)
        res = self.log_softmax(res)
        if encoding:
            return res, hidden_layer
        return res


# In[4]:


batch_size = 1000
file_name="../abc_full"
window_size=4
min_freq=10
batch_log_num = 10

idx_pairs, (word2idx, idx2word, vocabulary_size) = get_corpus_details(file_name=file_name, window_size=window_size, min_freq=min_freq)
dataset = PairDataset(idx_pairs, word2idx, idx2word)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(idx_pairs)
# print(word2idx)


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkipGram(vocab_size=vocabulary_size+1, hidden_size=100, device=device).to(device)
print(device)


# In[6]:


loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


# words_in_graph = ['Moon', 'Sun', 'French', 'Australia', 'directly', 'biological']
# words_in_graph = ['astronomer', 'Sun.', 'Sun\'s', 'Moon', 'Observatory', 'French', 'planet', 'coffee', 'Technology', 'sperm', 'nanotechnology', 'nanomaterials', 'shield', 'chimpanzees']
# for word in words_in_graph:
#     print(word, word2idx[word])

# In[ ]:

model_save_file_path = "model_win{}_minfreq{}.pth".format(window_size, min_freq)
plot_save_file_path = "plot_win{}_minfreq{}".format(window_size, min_freq)
if not os.path.isdir(plot_save_file_path):
    os.mkdir(plot_save_file_path)

if load_from_checkpoint:
    checkpoint = torch.load(model_save_file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    labels = []
    label_vectors = []
    word_to_vec_map = {}
    
    print("\nRunning val. Calculating accuracy...\n")
    correct_context = 0
    total_loss = 0
    start_time = datetime.now()
    
    if full_load:
        with torch.no_grad():
            for batch_i, pair_batch in enumerate(dataloader):
                data = pair_batch[:, 0].to(device)
                target = pair_batch[:, 1].to(device)
                result, enc = model(data, encoding=True)
                loss = loss_function(result, target)
                total_loss += loss.cpu().item()
        #         for i in range(pair_batch.shape[0]):
        #             item_result = result_np[i]
        #             item_target = target_np[i]
        #             window = item_result.argsort()[::-1][:4]
        #             if item_target in window:
        #                 correct_context += 1
                sorted_result = torch.topk(result, k=window_size*2, dim=1, sorted=False).indices
                correct_context += torch.sum(torch.sum(sorted_result == target.view((-1, 1)), dim=1)).cpu().item()


                enc = enc.cpu()
                data = pair_batch[:, 0]

    #             print(data.shape, "data.shape", data[0].item())

                for i in range(len(data)):
                    if data[i].item() not in word_to_vec_map:
                        if data[i].item() not in idx2word:
                            raise Exception("sdfgsf")

                        word_to_vec_map[data[i].item()] = 1
                        labels.append(idx2word[data[i].item()])
                        label_vectors.append(enc[i].tolist())


                if (batch_i+1) % batch_log_num == 0 or (batch_i+1) % len(dataloader) == 0:
                    print("\r batch:{}/{} time:{} correct_context:{}".format(batch_i+1, len(dataloader), 
                                                          datetime.now()-start_time, correct_context), end="")
        print("")
        print("\nAccuracy: {}".format(correct_context/len(dataset)))
        print("Total Loss: {} Item Loss: {}".format(total_loss, total_loss/len(dataset)))
        print("\n\n\n")
    else:
        with torch.no_grad():
            for batch_i, word in enumerate(words_in_graph):
                labels.append(word)

                res, enc = model(torch.tensor([word2idx[word]]).to(device), encoding=True)

                label_vectors.append(enc.cpu()[0].tolist())
    
    label_vectors = np.array(label_vectors)
    
    print("label_vectors.shape", label_vectors.shape)
    
    plot_tsne(plot_save_file_path, label_vectors, labels, epoch="only_words")
    exit("Exitting...")
    
    
    

num_epochs = 30
prev_loss = 0
for epoch in range(num_epochs):
    print("\nEpoch: {}".format(epoch))
    total_loss = 0
    start_time = datetime.now()

    for i, pair_batch in enumerate(dataloader):
        optimizer.zero_grad()
        data = pair_batch[:, 0].to(device)
        target = pair_batch[:, 1].to(device)
        result = model(data)
        loss = loss_function(result, target)
        total_loss += loss.cpu().item()

        loss.backward()
        optimizer.step()
        if (i+1) % batch_log_num == 0 or (i+1) % len(dataloader) == 0:
            print("\r batch:{}/{} time:{}".format(i+1, len(dataloader), datetime.now()-start_time), end="")

    print("")
    print("Total Loss: {} Item Loss: {}".format(total_loss, total_loss/len(dataset)))
    
    if epoch == 0 or prev_loss > total_loss:
        prev_loss = total_loss
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_save_file_path)
        print("\nModel saved at {}".format(model_save_file_path))
    
    print("\nCalculating accuracy...\n")
    correct_context = 0
    total_loss = 0
    start_time = datetime.now()
    
    if epoch % 2 == 0:
        with torch.no_grad():

            labels = []
            label_vectors = []
            word_to_vec_map = {}

            for batch_i, pair_batch in enumerate(dataloader):
                data = pair_batch[:, 0].to(device)
                target = pair_batch[:, 1].to(device)
                result, enc = model(data, encoding=True)
                loss = loss_function(result, target)
                total_loss += loss.cpu().item()
        #         for i in range(pair_batch.shape[0]):
        #             item_result = result_np[i]
        #             item_target = target_np[i]
        #             window = item_result.argsort()[::-1][:4]
        #             if item_target in window:
        #                 correct_context += 1
                sorted_result = torch.topk(result, k=window_size*2, dim=1, sorted=False).indices
                correct_context += torch.sum(torch.sum(sorted_result == target.view((-1, 1)), dim=1)).cpu().item()

                enc = enc.cpu()
                data = pair_batch[:, 0]

    #             print(data.shape, "data.shape", data[0].item())

                for i in range(len(data)):
                    if data[i].item() not in word_to_vec_map:
                        if data[i].item() not in idx2word:
                            raise Exception("sdfgsf")

                        word_to_vec_map[data[i].item()] = 1
                        labels.append(idx2word[data[i].item()])
                        label_vectors.append(enc[i].tolist())

                if (batch_i+1) % batch_log_num == 0 or (batch_i+1) % len(dataloader) == 0:
                    print("\r batch:{}/{} time:{} correct_context:{}".format(batch_i+1, len(dataloader), 
                                                          datetime.now()-start_time, correct_context), end="")

            print("")
            print("\nAccuracy: {}".format(correct_context/len(dataset)))
            print("Total Loss: {} Item Loss: {}".format(total_loss, total_loss/len(dataset)))
            print("\n\n\n")

            label_vectors = np.array(label_vectors)    
            print("label_vectors.shape", label_vectors.shape)

            plot_tsne(plot_save_file_path, label_vectors, labels, epoch=epoch)
        
    
