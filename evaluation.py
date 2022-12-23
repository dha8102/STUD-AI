import torch
import numpy as np
from tqdm import tqdm
from util import scoring_softmax as scoring
import torch.nn.functional as F
from models.MLP import MLP_Classification


def calculate(logits, labels, data_cnt, save_path):
    eval_acc, precision, recall, f1, eval_bunja, eval_pre_bunmo, eval_rec_bunmo = scoring(logits, labels,'val')
    eval_acc /= data_cnt
    try:
        eval_pre = eval_bunja / eval_pre_bunmo
    except:
        eval_pre = 0
    try:
        eval_rec = eval_bunja / eval_rec_bunmo
    except:
        eval_rec = 0
    try:
        eval_f1 = 2 * (eval_pre * eval_rec) / (eval_pre + eval_rec)
    except:
        eval_f1 = 0
    print("eval_acc:", eval_acc)
    print("eval_precision:", eval_pre)
    print("eval_recall:", eval_rec)
    print("eval_f1:", eval_f1)

    return eval_pre, eval_rec, eval_f1, eval_acc


def evaluation(save_path, model, dataloader, device):
    model_state_dict = torch.load(save_path + '/model_best.pth.tar', map_location=device)['state_dict']


    model.load_state_dict(model_state_dict)
    model.eval()

    model.eval()
    eval_acc = 0
    data_cnt = 0
    eval_bunja = 0
    eval_pre_bunmo = 0
    eval_rec_bunmo = 0
    data_cnt = 0
    with open(save_path + '/test_result.txt', 'w') as f:
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='testing:'):
            model_input = batch[0].to(device)
            labels = batch[1].to(device)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    logits = model(model_input)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc, precision, recall, f1, bunja, pre_bunmo, rec_bunmo = scoring(logits, labels, 'val')
            eval_bunja += bunja
            eval_pre_bunmo += pre_bunmo
            eval_rec_bunmo += rec_bunmo
            eval_acc += acc
            data_cnt += len(labels)

        eval_acc /= data_cnt
        try:
            eval_pre = eval_bunja / eval_pre_bunmo
        except:
            eval_pre = 0
        try:
            eval_rec = eval_bunja / eval_rec_bunmo
        except:
            eval_rec = 0
        try:
            eval_f1 = 2 * (eval_pre * eval_rec) / (eval_pre + eval_rec)
        except:
            eval_f1 = 0

        print("=============================")
        print("eval_acc:", eval_acc)
        print("eval_precision:", eval_pre)
        print("eval_recall:", eval_rec)
        print("eval_f1:", eval_f1)


if __name__ == "__main__":
    folder_path = './saved_model_MLP/2022-10-26-15-21-18'
    test_file = './dataset_onehot/preprocessed_test_onehot_minmax_dummy.csv'

    device = torch.device("cuda")

    from torch.utils.data import DataLoader
    from data import Baselinedataset, collator

    model = MLP_Classification().to(device)
    test_dataset = Baselinedataset(test_file, None, 'MLP')
    eval_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    evaluation(folder_path, model, eval_dataloader, device)
