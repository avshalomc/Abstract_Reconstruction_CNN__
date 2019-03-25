from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sumeval.metrics.rouge import RougeCalculator
from hyperdash import Experiment
from torch.utils.data import DataLoader
import os
import visdom
import argparse
import math
from copy import deepcopy


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

class DeconvolutionDecoder(nn.Module):
    def __init__(self, embedding, tau, sentence_len, filter_size, filter_shape, latent_size):
        super(DeconvolutionDecoder, self).__init__()
        self.tau = tau
        self.embed = embedding
        self.deconvs1 = nn.ConvTranspose2d(latent_size, filter_size * 2, (sentence_len, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(filter_size * 2)
        self.deconvs2 = nn.ConvTranspose2d(filter_size * 2, filter_size, (filter_shape, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(filter_size)
        self.deconvs3 = nn.ConvTranspose2d(filter_size, 1, (filter_shape, self.embed.weight.size()[1]), stride=2)

        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def __call__(self, h3):
        h2 = F.relu(self.bn1(self.deconvs1(h3)))
        h1 = F.relu(self.bn2(self.deconvs2(h2)))
        x_hat = F.relu(self.deconvs3(h1))
        x_hat = x_hat.squeeze()

        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x_hat.size()) < 3:
            x_hat = x_hat.view(1, *x_hat.size())
        # normalize
        norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
        rec_x_hat = x_hat / norm_x_hat

        # compute probability
        norm_w = self.embed.weight.data.t()
        prob_logits = torch.bmm(rec_x_hat, norm_w.unsqueeze(0)
                         .expand(rec_x_hat.size(0), *norm_w.size())) / self.tau
        log_prob = F.log_softmax(prob_logits, dim=2)
        return log_prob

def transform_id2word(index, id2word, lang):
    if lang == "ja":
        return "".join([id2word[idx.item()] for idx in index])
    else:
        return " ".join([id2word[idx.item()] for idx in index])

def sigmoid_annealing_schedule(step, max_step, param_init=1.0, param_final=0.01, gain=0.3):
    return ((param_init - param_final) / (1 + math.exp(gain * (step - (max_step / 2))))) + param_final

def save_models(model, path, prefix, steps):
    if not os.path.isdir(path):
        os.makedirs(path)
    model_save_path = '{}/{}_steps_{}.pt'.format(path, prefix, steps)
    torch.save(model, model_save_path)

def calc_rouge(original_sentences, predict_sentences):
    rouge_1 = 0.0
    rouge_2 = 0.0
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        rouge = RougeCalculator(stopwords=True, lang="en")
        r1 = rouge.rouge_1(summary=predict, references=original)
        r2 = rouge.rouge_2(summary=predict, references=original)
        rouge_1 += r1
        rouge_2 += r2
    return rouge_1, rouge_2

def eval_reconstruction(encoder, decoder, data_iter, args, vis, win, epoch):
    print("=================Eval======================")
    encoder.eval()
    decoder.eval()
    avg_loss = 0
    rouge_1 = 0.0
    rouge_2 = 0.0
    index2word = data_iter.dataset.index2word
    epoch_eval_losses = []
    for batch in data_iter:
        feature = batch #requires_grad=False)
        if args.use_cuda:
            feature = feature.cuda()
        h = encoder(feature)
        prob = decoder(h)
        _, predict_index = torch.max(prob, 2)
        original_sentences = [transform_id2word(sentence, index2word, "en") for sentence in batch]
        predict_sentences = [transform_id2word(sentence, index2word, "en") for sentence in predict_index.data]
        r1, r2 = calc_rouge(original_sentences, predict_sentences)
        rouge_1 += r1
        rouge_2 += r2
        reconstruction_loss = compute_cross_entropy(prob, feature)
        avg_loss += reconstruction_loss.item()
        # Visualization data
        epoch_eval_losses.append(reconstruction_loss.item())

    # visualisation
    epoch_eval_loss = sum(epoch_eval_losses) / float(len(epoch_eval_losses))
    vis.line(X=np.array((epoch,)), Y=np.array((epoch_eval_loss,)), name="eval_loss", update="append", win=win)
    #epoch_eval_losses.clear()

    #print(original_sentences[0])
    #print(predict_sentences[0])

    avg_loss = avg_loss / len(data_iter.dataset)
    avg_loss = avg_loss / args.sentence_len
    rouge_1 = rouge_1 / len(data_iter.dataset)
    rouge_2 = rouge_2 / len(data_iter.dataset)
    print("Evaluation - loss: {}  Rouge1: {}    Rouge2: {}".format(avg_loss, rouge_1, rouge_2))
    print("===============================================================")
    encoder.train()
    decoder.train()

def compute_cross_entropy(log_prob, target):
    # compute reconstruction loss using cross entropy
    loss = [F.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, target)]
    average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]
    return average_loss

def train_reconstruction(train_loader, test_loader, encoder, decoder, args):
    exp = Experiment("Reconstruction Training")
    #vis = Visualizations()
    vis = visdom.Visdom(port=8098)
    try:
        lr = args.lr
        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
        decoder_opt = torch.optim.Adam(decoder.parameters(), lr=lr)

        encoder.train()
        decoder.train()
        steps = 0
        all_losses = []
        for epoch in range(1, args.epochs+1):
            epoch_losses = []
            print("=======Epoch========")
            print(epoch)
            for batch in train_loader:
                feature = batch # Variable
                if args.use_cuda:
                    encoder.cuda()
                    decoder.cuda()
                    feature = feature.cuda()

                encoder_opt.zero_grad()
                decoder_opt.zero_grad()

                h = encoder(feature)
                prob = decoder(h)
                reconstruction_loss = compute_cross_entropy(prob, feature)
                reconstruction_loss.backward()
                encoder_opt.step()
                decoder_opt.step()

                print("Epoch: {}".format(epoch))
                print("Steps: {}".format(steps))
                print("Loss: {}".format(reconstruction_loss.item() / args.sentence_len))
                exp.metric("Loss", reconstruction_loss.item() / args.sentence_len)

                epoch_losses.append(reconstruction_loss.item())

                # check reconstructed sentence
                if steps % args.log_interval == 0:
                    print("Test!!")
                    input_data = feature[0]
                    single_data = prob[0]
                    _, predict_index = torch.max(single_data, 1)
                    input_sentence = transform_id2word(input_data.data, train_loader.dataset.index2word, lang="en")
                    predict_sentence = transform_id2word(predict_index.data, train_loader.dataset.index2word, lang="en")
                    print("Input Sentence:")
                    print(input_sentence)
                    print("Output Sentence:")
                    print(predict_sentence)

                steps += 1

            # Visualization data

            epoch_loss = sum(epoch_losses) / float(len(epoch_losses))
            all_losses.append(epoch_loss)
            if epoch == 1:
                # vis.plot_loss(np.mean(epoch_losses), steps)
                win = vis.line(X=np.array((epoch,)), Y=np.array((epoch_loss,)), name="train_loss", opts=dict(xlabel='Epoch',ylabel='Loss', title='Train and Eval Loss'))
            else:
                vis.line(X=np.array((epoch,)), Y=np.array((epoch_loss,)), name="train_loss", update="append", win=win)
            #epoch_losses.clear()

            if epoch % args.test_interval == 0:
                eval_reconstruction(encoder, decoder, test_loader, args, vis, win, epoch)


            if epoch % args.lr_decay_interval == 0:
                # decrease learning rate
                lr = lr / 1.05
                encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
                decoder_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
                encoder.train()
                decoder.train()

            if epoch % args.save_interval == 0:
                save_models(encoder, args.save_dir, "encoder", steps)
                save_models(decoder, args.save_dir, "decoder", steps)

            if epoch % 20 == 0:
            # finalization
            # save vocabulary
            #with open("word2index", "wb") as w2i, open("index2word", "wb") as i2w:
            #    pickle.dump(train_loader.dataset.word2index, w2i)
            #    pickle.dump(train_loader.dataset.index2word, i2w)
                torch.save(train_loader.dataset.index2word, "/home/avshalom/ext/ae_cnn_code/index2word.pt")
                torch.save(train_loader.dataset.word2index, "/home/avshalom/ext/ae_cnn_code/word2index.pt")

            # save models
            #save_models(encoder, args.save_dir, "encoder", "final")
            #save_models(decoder, args.save_dir, "decoder", "final")
                torch.save(encoder, "/home/avshalom/ext/ae_cnn_code/encoder_lsize_%s_epoch_%s.pt" % (args.latent_size, epoch))

        print("Finish!!!")
    finally:
        exp.end()

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

def main():
    parser = argparse.ArgumentParser(description='text convolution-deconvolution auto-encoder model')
    # learning
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate') # 0.001, 0.01, 0.1
    parser.add_argument('-epochs', type=int, default=500, help='number of epochs for train')
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

    # (1) original data:
    # train_data, test_data = load_hotel_review_data(args.data_path, args.sentence_len)
    # (2) small original data:
    # train_data, test_data = torch.load('/home/avshalom/ext/ae_cnn_data/hotel_reviews_small_train.pt'), torch.load('/home/avshalom/ext/ae_cnn_data/hotel_reviews_small_test.pt')
    # (3) abstracts data
    train_data, test_data = torch.load('/home/avshalom/ext/ae_cnn_data/train_abstracts.pt'), torch.load('/home/avshalom/ext/ae_cnn_data/test_abstracts.pt')


    train_loader, test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle),\
                                  DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

    k = args.embed_dim
    v = train_data.vocab_lennght()
    t1 = args.sentence_len + 2 * (args.filter_shape - 1)
    t2 = int(math.floor((t1 - args.filter_shape) / 2) + 1) # "2" means stride size
    t3 = int(math.floor((t2 - args.filter_shape) / 2) + 1) - 2
    if args.enc_snapshot is None or args.dec_snapshot is None:
        print("Start from initial")
        embedding = nn.Embedding(v, k, max_norm=1.0, norm_type=2.0)

        encoder = ConvolutionEncoder(embedding, t3, args.filter_size, args.filter_shape, args.latent_size)
        decoder = DeconvolutionDecoder(embedding, args.tau, t3, args.filter_size, args.filter_shape, args.latent_size)
    else:
        print("Restart from snapshot")
        encoder = torch.load(args.enc_snapshot)
        decoder = torch.load(args.dec_snapshot)

    train_reconstruction(train_loader, test_loader, encoder, decoder, args)

if __name__ == '__main__':
    print("")
    main()