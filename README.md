# NeuralNetwork

This is the primary project I have been working on for the past few months. I am building linear, convolutional and transformer models (both encoder-only and encoder-decoder ones) and the entire infrastructure for tokenizing and encoding-decoding text, processing data, constructing dataloaders, training the network, monitoring performance and saving networks. 

I include three notebooks where I:

1) train an efficient convolutional network on the FashionMNIST database mainly for fun,

2) train a medium-sized (nlayer=2, d_model = 600) transfromer on Wikitext-2 whose performance compares well with the pytorch built-in model

3) use my transformer model to investigate the scaling laws of neural network performance. Consistently with the observations in the literature, I find a power-law decay of the minimal loss (measured at convergence) as a function of the network size at fixed dataset size, until performance gets bottlenecked by the size of the dataset and the minimal loss plateaus. I also investigate the effect of increasing width, depth or both while keeping the aspect ratio fixed and find that fixed depth = 2 layers while increasing width is the optimal direction for increasing model size in this example.

A few more projects are in order, including a more in-depth exploration of the scaling laws (i.e. training models at fixed compute), an investigation of the grokking phenomenon in modular arithmetic and the phenomenon of feature superposition and its signatures in the training loss and the effective number of dimensions per feature in the hidden layer representation of dense non-linear network.
