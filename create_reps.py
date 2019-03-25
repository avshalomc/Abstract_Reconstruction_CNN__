import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import listdir
from os.path import isfile, join
from nltk.tokenize import TweetTokenizer
import torch
import copy
import argparse
import math
from copy import deepcopy

class ConvolutionEncoder(nn.Module):
    def __init__(self, embedding, sentence_len, filter_size, filter_shape, latent_size):
        super(ConvolutionEncoder, self).__init__()
        self.embed = embedding
        self.convs1 = nn.Conv2d(1, filter_size, (filter_shape, self.embed.weight.size()[1]), stride=2)
        self.bn1 = nn.BatchNorm2d(filter_size)
        self.convs2 = nn.Conv2d(filter_size, filter_size * 2, (filter_shape, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(filter_size * 2)
        self.convs3 = nn.Conv2d(filter_size * 2, latent_size, (sentence_len, 1), stride=2)

        # weight initialize for conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def __call__(self, x):
        x = self.embed(x)

        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x.size()) < 3:
            x = x.view(1, *x.size())
        # reshape for convolution layer
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])

        h1 = F.relu(self.bn1(self.convs1(x)))
        h2 = F.relu(self.bn2(self.convs2(h1)))
        h3 = F.relu(self.convs3(h2))

        return h3

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

    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate') # 0.001, 0.01, 0.1
    parser.add_argument('-epochs', type=int, default=400, help='number of epochs for train')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('-lr_decay_interval', type=int, default=4,
                        help='how many epochs to wait before decrease learning rate')
    parser.add_argument('-log_interval', type=int, default=256,
                        help='how many steps to wait before logging training status')
    parser.add_argument('-test_interval', type=int, default=1,
                        help='how many epochs to wait before testing')
    parser.add_argument('-save_interval', type=int, default=100,
                        help='how many epochs to wait before saving')
    parser.add_argument('-save_dir', type=str, default='rec_snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-data_path', default='/home/avshalom/ext/ae_cnn_data/hotel_reviews.p', type=str, help='data path')
    parser.add_argument('-shuffle', default=False, help='shuffle data every epoch')
    parser.add_argument('-sentence_len', type=int, default=253, help='how many tokens in a sentence')
    # model
    parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension')
    parser.add_argument('-filter_size', type=int, default=300, help='filter size of convolution')
    parser.add_argument('-filter_shape', type=int, default=5,
                        help='filter shape to use for convolution')
    parser.add_argument('-latent_size', type=int, default=900, help='size of latent variable') #900/50
    parser.add_argument('-tau', type=float, default=0.01, help='temperature parameter')
    parser.add_argument('-use_cuda', action='store_true', default=True, help='whether using cuda')
    # option
    parser.add_argument('-enc_snapshot', type=str, default=None, help='filename of encoder snapshot ')
    parser.add_argument('-dec_snapshot', type=str, default=None, help='filename of decoder snapshot ')
    args = parser.parse_args()

    index2word = torch.load("/home/avshalom/ext/ae_cnn_code/index2word.pt")
    word2index = torch.load("/home/avshalom/ext/ae_cnn_code/word2index.pt")

    encoder = torch.load("/home/avshalom/ext/ae_cnn_code/encoder_lsize_900_epoch_500.pt")
    encoder.eval()
    encoder.cuda()

    categories = ['eco', 'soc', 'med', 'psy']

    for category in categories:
        args.data_dir = "/home/avshalom/ext/ae_cnn_data/%s_abstracts/" % category
        data_ndarray = create_data_ndarray(args, index2word, word2index) # takes time...
        size = data_ndarray.shape[0]
        sizek = int(size/1000) + 1
        list_of_latent_chunks = []
        for i in range(sizek):
            if i==sizek-1:
                chunk = data_ndarray[i*1000:]
            else:
                chunk = data_ndarray[1000*i:1000*(i+1)]
            latent_chunk = encoder(torch.tensor(chunk).cuda()).squeeze(-1).squeeze(-1)
            device = torch.device('cpu')
            #latent_chunk = latent_chunk.to(device)
            list_of_latent_chunks.append(copy.copy(latent_chunk.to(device)))
            del latent_chunk
        latent = torch.cat(list_of_latent_chunks)
        torch.save(latent, "/home/avshalom/ext/ae_cnn_data/%s_%s_latent.pt" % (category,str(args.latent_size)))
        del latent, list_of_latent_chunks


if __name__ == '__main__':
    main()
