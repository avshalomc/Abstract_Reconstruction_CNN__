import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
from nltk.tokenize import TweetTokenizer
import operator
#import random
import torch
from torch.utils.data import Dataset
import copy


class HotelReviewsDataset(Dataset):
    """
    Hotel Reviews Dataset
    """

    def __init__(self, data_list, word2index, index2word, sentence_len, transform=None):
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = len(self.word2index)
        self.data = data_list
        self.sentence_len = sentence_len
        self.transform = transform
        self.word2index["<PAD>"] = self.n_words
        self.index2word[self.n_words] = "<PAD>"
        self.n_words += 1
        temp_list = []
        for sentence in tqdm(self.data):
            if len(sentence) > self.sentence_len:
                # truncate sentence if sentence length is longer than `sentence_len`
                temp_list.append(np.array(sentence[:self.sentence_len]))
            else:
                # pad sentence  with '<PAD>' token if sentence length is shorter than `sentence_len`
                sent_array = np.lib.pad(np.array(sentence),
                                        (0, self.sentence_len - len(sentence)),
                                        "constant",
                                        constant_values=(self.n_words - 1, self.n_words - 1))
                temp_list.append(sent_array)
        self.data = np.array(temp_list, dtype=np.int32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def vocab_lennght(self):
        return len(self.word2index)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data).type(torch.LongTensor)


def load_hotel_review_data(path, sentence_len):
    """
    Load Hotel Reviews data from pickle distributed in https://drive.google.com/file/d/0B52eYWrYWqIpQzhBNkVxaV9mMjQ/view
    This file is published in https://github.com/dreasysnail/textCNN_public

    :param path: pickle path
    :return:
    """
    import _pickle as cPickle
    with open(path, "rb") as f:
        data = cPickle.load(f, encoding="latin1")

    train_data, test_data = HotelReviewsDataset(data[0], deepcopy(data[2]), deepcopy(data[3]), sentence_len,
                                                transform=ToTensor()), \
                            HotelReviewsDataset(data[1], deepcopy(data[2]), deepcopy(data[3]), sentence_len,
                                                transform=ToTensor())
    return train_data, test_data


def create_vocab(args):
    index2word = {}
    word2index = {}

    wordcount = {}

    list_of_abs_files = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

    tknzr = TweetTokenizer()

    for file in list_of_abs_files:
        filename = args.data_dir + file
        file = open(filename, 'rt')
        text = file.read()
        file.close()
        tokens = tknzr.tokenize(text)

        for token in tokens:
            if token in wordcount.keys():
                wordcount[token] += 1
            else:
                wordcount[token] = 1

    # sort by count
    sorted_wordcount = sorted(wordcount.items(), key=operator.itemgetter(1))
    sorted_wordcount.reverse()

    # prune vocab
    if len(sorted_wordcount) > args.vocab_size:
        sorted_wordcount = sorted_wordcount[:args.vocab_size]

    # special tokens
    index2word[0] = 'END_TOKEN'
    word2index['END_TOKEN'] = 0

    index2word[1] = 'UNknown'
    word2index['UNknown'] = 1

    index2word[args.vocab_size+2] = '<PAD>' # 25002
    word2index['<PAD>'] = args.vocab_size+2 # 25002

    index = 2
    for pair in sorted_wordcount:
        index2word[index] = pair[0]
        word2index[pair[0]] = index
        index += 1

    #print(sorted_wordcount)
    #print(index2word)
    #print(word2index)

    return index2word, word2index


def create_data_ndarray(args, index2word, word2index):

    N = args.txt_len # 253
    list_of_abs_files = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]
    #list_of_abs_files = random.shuffle(list_of_abs_files)

    data = np.zeros([len(list_of_abs_files),N], dtype=int)  # [datalen X 253]

    tknzr = TweetTokenizer()

    for i, file in enumerate(list_of_abs_files):
        filename = args.data_dir + file
        file = open(filename, 'rt')
        text = file.read()
        file.close()
        tokens = tknzr.tokenize(text)

        if len(tokens) > N:
            tokens = tokens[:N]
        else:
            tokens += ['<UNknown>']*(N-len(tokens))

        indices = np.array([int(word2index[token]) if token in word2index.keys() else 25002 for token in tokens])

        data[i] = indices

    return data


def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    parser.add_argument('-txt_len', type=int, default=253, help='initial learning rate')
    parser.add_argument('-vocab_size', type=int, default=25000, help='number of words in vocab')
    parser.add_argument('-data_dir', type=str, default='/home/avshalom/ext/ae_cnn_data/abstracts/', help='data directory containing all txt files')
    parser.add_argument('-save_path_train', type=str, default='/home/avshalom/ext/ae_cnn_data/train_abstracts.pt', help='data directory containing all txt files')
    parser.add_argument('-save_path_test', type=str, default='/home/avshalom/ext/ae_cnn_data/test_abstracts.pt', help='data directory containing all txt files')
    parser.add_argument('-test_data', type=float, default=0.3, help='presentage for test data')
    args = parser.parse_args()

    index2word, word2index = create_vocab(args)

    data = create_data_ndarray(args, index2word, word2index)

    # TODO: load original reviews data
    hotel_data = torch.load('/home/avshalom/ext/ae_cnn_data/hotel_reviews_small_train.pt')
    print("")

    # TODO: replace original data and vocab with abstracts and save it
    new_data = hotel_data
    new_data.data = data
    new_data.index2word = index2word
    new_data.word2index = word2index
    new_data.n_words = args.vocab_size+2
    new_data.sentence_len = args.txt_len

    test_size = int(data.shape[0]*args.test_data)
    train_size = data.shape[0]-test_size

    new_train_data = copy.copy(new_data)
    new_train_data.data = new_data.data[:train_size]

    new_test_data = copy.copy(new_data)
    new_test_data.data = new_data.data[-test_size:]

    torch.save(new_train_data, args.save_path_train)
    torch.save(new_test_data, args.save_path_test)

if __name__ == '__main__':
    main()