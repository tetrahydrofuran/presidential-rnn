import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from .basernn import  baseRNN
from matplotlib import pyplot as plt
import logging

# Global variables
char_to_index = {}
index_to_char = {}
data_size = 0
char_size = 0

model = None
optimizer = None
criterion = None

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005
chunk_len = 200

parser = argparse.ArgumentParser()
# TODO: Argument parser


def train(corpus, plot_loss=True, label='chunk'):
    cuda = True if torch.cuda.is_available() else False
    # TODO implement CUDA
    global model
    global optimizer
    global criterion

    # TODO implement model selection
    model = baseRNN(char_size, hidden_size, char_size, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    all_losses = []
    loss_avg = 0

    for epoch in range(1, n_epochs + 1):
        inp, target = rand_train(corpus)

        loss = train_step(inp, target)
        loss_avg += loss

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_step(start), epoch, epoch / n_epochs * 100, loss))
            print(evaluate(primer='Wh', sample_len=100), '\n')

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

    if plot_loss:
        plot(all_losses)
    dump(model, label + '.model')


def train_step(inp, target):
    hidden = model.init_hidden()
    model.zero_grad()  # zero gradient
    loss = 0

    for c in range(chunk_len):
        output, hidden = model(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    optimizer.step()
    return loss.data[0] / chunk_len


def evaluate(primer='A', sample_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = char_tensor(primer)
    predicted = primer

    # build up the hidden state
    for p in range(len(primer) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(sample_len):
        output, hidden = model(inp, hidden)

        # Sample multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        # Select one value from multinomial distribution, first index (value)
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add character to input
        predicted_character = index_to_char[top_i.item()]  # validate
        predicted += predicted_character
        inp = char_tensor(predicted_character)

    return predicted


# region Model Helpers
def map_corpus(corpus):
    """
    Creates set of characters, and creates mapping of unique characters to integer and inverse, assigned
    to global scope
    :param corpus:
    :return:
    """
    characters = list(set(corpus))

    global char_to_index
    global index_to_char
    global data_size
    global char_size

    data_size = len(corpus)
    char_size = len(characters)
    char_to_index = {ch: i for i, ch in enumerate(characters)}
    index_to_char = {i: ch for i, ch in enumerate(characters)}
    logging.debug('map_corpus(): Data size: ' + str(data_size))
    logging.debug('map_corpus(): Char size: ' + str(char_size))


def time_step(since):
    s = time.time() - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


def plot(loss):
    # TODO xticks, yticks, title
    plt.figure()
    plt.plot(loss)
    plt.show()


def dump(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    pass
# TODO
# endregion


# region Torch Model Prep
def random_chunk(corpus, chunksize=200):
    start = np.random.randint(0, data_size - chunksize)
    return corpus[start:start + chunksize + 1]


def char_tensor(charstring):
    tensor = torch.zeros(len(charstring)).long()
    for i, ch in enumerate(charstring):
        tensor[i] = char_to_index[ch]
    return tensor


def rand_train(corpus, chunk_size=200):
    chunk = random_chunk(corpus, chunk_size)
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

# endregion
