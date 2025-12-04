基于 BERT 中文电子病历命名实体识别---采用基于 BERT 的融合模型，对医疗公共数据集 CCKS2019 进行命名实体识别研究，模型在测试集上达到了 0.860 的 F1 分数。

一. 运行说明：

1.1 实验环境

GPU—A2000; CPU—6x Xeon E5-2680 v4; 内存—30G; HuggingFace系统镜像；Pytorch 3.8；CUDA 11.3；Python 3.8；transformer 3.4；protobuf 3.19.0;tensorflow； pytorch-crf等

1.2 运行方式

在终端中运行代码：python main.py 

若报错TypeError: Descriptors cannot be created directly. If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0. If you cannot immediately regenerate your protos, some other possible workarounds are: 
1. Downgrade the protobuf package to 3.20.x or lower.
2. Set PROTOCOL BUFFERS PYTHON IMPLEMENTATION=python (but this will use pure Python parsing and will be much slower)

则需要降低 protobuf 版本，并在 BERT+Bi_LSTM+CRF.ipynb 中通过 pip install protobuf==3.19.0 运行代码。

二. 模型描述

2.1 数据预处理：对文本进行按段进行切割；BIO 序列标注

2.2 数据集构建：定义数据集类（NerDataset），将文本数据转换为适合 PyTorch 框架输入的格式。使用句号作为分隔符，从处理后的文本中顺序读取序列。最大序列长度设置为 256。当序列长度超过最大值时，将其截断到最大长度，并在存储前在序列开头和结尾添加标记（[‘CLS’], [‘SEP’]）。同时，为了确保批次中每个样本的序列长度（batch）一致，设置了序列填充函数（PadBatch）。当确定该样本的序列长度小于 256 时，用零将样本序列填充至 256。

2.3 数据加载：使用PyTorch平台的数据加载类（DateLoader）创建了三个数据迭代器：train_iter、eval_iter和test_iter，分别在训练、验证和测试时加载批量数据。其中参数dataset表示输入的数据集。参数batch_size表示每个batch中的样本数量，训练时设置为64，验证和测试时设置为32。每个epoch都会将数据顺序打乱，以避免模型学习到数据顺序。数据加载时的子进程数据设置为4，可以降低数据读取时间。加载过程中对样本的填充方式为PadBatch

三. Bert-Base-Chinese + Bi-LSTM + CRF模型构建

3.1 BERT模型接受输入参数（sentence）,先将输入的字符映射为ID，ID经过编码层（Embedding Layer）得到char embedding，其维度为768，将其存入embeds中并送入Bi-LSTM层

3.2 Bi-LSTM模型的输入维度为768，层数设置为2。Bi-LSTM模型将对embeds进行特征提取，得到上下文编码enc

3.3 enc经过Dropout层正则化后，通过线性层映射到标签空间，得到发射矩阵emissions并传入CRF层。训练时，CRF将计算负对数似然损失，用于训练优化，优化算法为Adam；测试时，CRF将通过解码获得预测的最佳标签序列

四. 模型优化

4.1 学习率优化

选择线性预热（Linear Warmup）策略进行学习率优化。选择该策略能在训练初期，平稳增加学习率，以避免跳过损失函数的有效区域，有利于模型收敛。
具体来说，使用了调度器函数（get_linear_schedule_with_warmup），根据预热步数（num_warmup_steps）和总训练步数（total_steps）自动调整学习率。预热期间，学习率将从0线性增加优化器（AdamW）中设置的初始化学习率lr（值为0.001）。之后学习率将再线性降低到0

4.2 正则化

选择随机失活（Dropout）对模型进行正则化。训练时，随机将某些神经元输出归零，增强本实验模型的鲁棒性，从实验结果来看，在一定程度上避免了过拟合。
