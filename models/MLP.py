import torch
import torch.nn as nn

# sklearn mlp

# class MLP_Classification(nn.Module):
#
#     def __init__(self):
#         super(MLP_Classification, self).__init__()
#
#         self.mlp = nn.Sequential(
#             nn.Linear(1952, 100),
#             nn.ReLU(inplace=True),
#             nn.Linear(100, 20),
#             nn.ReLU(inplace=True),
#             nn.Linear(20, 2),
#         )
#
#     def forward(self, inp):
#         x = self.mlp(inp)
#         return x

# # original mlp
class MLP_Classification(nn.Module):

    def __init__(self):
        super(MLP_Classification, self).__init__()
        p = 0.3
        self.mlp = nn.Sequential(
            nn.Linear(1952, 1000),
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
            nn.Linear(1000, 2)
        )

    def forward(self, inp):
        x = self.mlp(inp)
        return x
