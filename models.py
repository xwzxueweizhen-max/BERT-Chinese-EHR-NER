import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2,
                            num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, sentence):
        #本机跑时，服务器上可以去掉，可以更新bert参数
        with torch.no_grad():
            embeds, _  = self.bert(sentence)
        #embeds, _  = self.bert(sentence)
        enc, _ = self.lstm(embeds)
        enc = self.dropout(enc)
        feats = self.linear(enc)
        return feats
    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test: # 训练时，返回损失
            loss=-self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else: # 测试时，返回decoder
            decode=self.crf.decode(emissions, mask)
            return decode

"""
#去掉bi-lstm层的模型架构，为了方便调用，未改变类名
    
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, embedding_dim=768, hidden_dim=256):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(embedding_dim, self.tagset_size)  # linear层输入维度改为embedding_dim
        self.crf = CRF(self.tagset_size, batch_first=True)
    
    def _get_features(self, sentence):
        with torch.no_grad():
            embeds, _ = self.bert(sentence)
        
        # 去掉bilstm层后的处理
        enc = self.dropout(embeds)
        feats = self.linear(enc)
        return feats

    def forward(self, sentence, tags, mask, is_test=False):
        emissions = self._get_features(sentence)
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(emissions, tags, mask, reduction='mean')
            return loss
        else:  # Testing，return decoding
            decode = self.crf.decode(emissions, mask)
            return decode
"""