import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import transformers.models.bert.modeling_bert


class BERT_MLPClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_MLPClassification, self).__init__(config)
        self.bert = BertModel(config)
        p = 0.3
        self.mlp = nn.Sequential(
            nn.Linear(1933, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 768)
        )
        self.f_mlp = nn.Sequential(
            nn.Linear(2* 768, 300),
            nn.Linear(300, 2)
        )

    def forward(self, model_input, input_ids, att_mask):
        # model_input: [batch_size, 1952]
        # input_ids, attention_mask : [batch_size,  maxlen(512)]

        bert_outputs = self.bert(input_ids, attention_mask=att_mask)
        pooled_output = bert_outputs.pooler_output  # [batch_size, 768]
        mlp_outputs = self.mlp(model_input)  # [batch_size, 768]
        cat = torch.cat((pooled_output, mlp_outputs), dim=1)
        return self.f_mlp(cat)
