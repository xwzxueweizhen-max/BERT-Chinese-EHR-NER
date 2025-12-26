Named Entity Recognition for Chinese Electronic Medical Records Based on BERT :Bert-Base-Chinese + Bi-LSTM + CRF

1. Running Instructions

1.1 Experimental Environment

GPU—A2000; CPU—6x Xeon E5-2680 v4; Memory—30G; HuggingFace；CUDA 11.3；Python 3.8；transformer 3.4；protobuf 3.19.0;tensorflow； pytorch-crf

1.2 Execution Method

bash：python main.py 

1.3 Error Handling
TypeError: Descriptors cannot be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. If you cannot immediately regenerate your protos, some other possible workarounds are: 
1. Downgrade the protobuf package to 3.20.x or lower.
2. Set PROTOCOL BUFFERS PYTHON IMPLEMENTATION=python (but this will use pure Python parsing and will be much slower)

Resolve it by downgrading the protobuf version and running the following command in the BERT+Bi_LSTM+CRF.ipynb notebook： pip install protobuf==3.19.0 

2. Model Description

2.1 Sequence labeling：BIO labeling scheme 

2.2 Dataset Construction：

2.2.1 Define NerDataset to convert raw text data into PyTorch-compatible tensor forma

2.2.2 Extract sequences from processed text sequentially using Chinese periods (。) as delimiters

2.2.3 Set maximum sequence length to 256 tokens；Add special tokens (['CLS'] at the start, [SEP'] at the end) to all sequences before storage；Implement a PadBatch function to ensure uniform sequence length across a batch—pad sequences shorter than 256 tokens with zeros to reach the 256-length standard.

2.3 Data Loading

2.3.1 Use PyTorch's DataLoader to create three iterators (train_iter, eval_iter, test_iter) for training, validation, and testing respectively.

2.3.2 Key DataLoader parameters

batch_size: 64 for training, 32 for validation/testing.

Use 4 subprocesses to accelerate data reading.

三. Model Construction

3.1 Input processing: The BERT model takes sentence as input, maps each character to a unique ID via the BERT vocabulary.Embedding generation: The character IDs are fed into the Embedding Layer to generate 768-dimensional character embeddings (char embedding). Store the embeddings in the embeds tensor and pass it to the Bi-LSTM layer.

3.2 Layer configuration: Bi-LSTM is initialized with an input dimension of 768 (matching BERT embeddings) and 2 hidden layers.Contextual encoding: The Bi-LSTM processes embeds to capture bidirectional contextual features, outputting contextual encodings (enc).

3.3 Apply Dropout regularization to enc to prevent overfitting.Map the regularized encodings to the label space via a linear layer to generate the emission matrix (emissions).Training phase: The CRF layer computes the negative log-likelihood loss (NLLLoss) to optimize model parameters, with the Adam optimizer used for gradient descent.Testing phase: The CRF layer performs Viterbi decoding on emissions to generate the optimal predicted label sequence (maximizing the conditional probability of the label sequence given the input).


