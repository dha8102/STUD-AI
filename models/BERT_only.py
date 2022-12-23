import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import transformers.models.bert.modeling_bert


class BERT_onlyClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BERT_onlyClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.f_mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(200, 2)
        )

    def forward(self, model_input, input_ids, att_mask):
        # model_input: [batch_size, 1952]
        # input_ids, attention_mask : [batch_size,  maxlen(512)]
        bert_outputs = self.bert(input_ids, attention_mask=att_mask)
        pooled_output = bert_outputs.pooler_output  # [batch_size, 768]
        return self.f_mlp(pooled_output)

