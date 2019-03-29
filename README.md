# Abstract_Reconstruction_CNN__

## BACKGROUND

This repository contains code that trains an autoencoder for article abstracts. 

It is based on the repository of the article "Deconvolutional Paragraph Representation Learning"

The articles are taken from famouse jornals in psychology, economics, sociality and medicine.

## CODE DETAILS

preprocess.py - tokenising 47K abstracts, creating 25k size vocabulary, saving the vocab and articles as ndarrays.

ae.py - train and eval CNN autoencoder

create_reps.py - after the models are trained, creating 200 and 900 size latent representations.

## OTHER

.svg files - loss and eval graphs. ae_30k_50.svg is a file that shows that 50 size latent space is too small.
index2word, word2index - are the 2 sided vocabs created during preprocess.
