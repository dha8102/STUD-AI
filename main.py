import torch
import time
import os
import argparse
import torch.nn as nn
import numpy as np
import random
import pickle
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup as linear_lr
from transformers import AutoTokenizer
from config import Configuration
from data import Baselinedataset, collator
from models.BERT_MLP import BERT_MLPClassification
from models.MLP import MLP_Classification
from models.BERT_only import BERT_onlyClassification
from util import scoring_softmax as scoring
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

    # configuration
    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=6, help='')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--sampling_weight', type=float, default=0, help='0 is not use, -1 for real sample ratio')
    parser.add_argument('--class_weight', type=float, default=0, help='0 is not use, -1 for real sample ratio')
    parser.add_argument('--maxlen', type=int, default=512)
    parser.add_argument('--model', type=str, default='MLP', help='MLP or BERT_MLP')
    parser.add_argument('--acc_steps', type=int, default=36, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    args = parser.parse_args()
    if args.model == 'BERT_MLP':
        folder_path = './saved_model_BERT_MLP/'
    elif args.model == 'BERT_only':
        folder_path = './saved_model_BERT_only/'
    elif args.model == 'MLP':
        folder_path = './saved_model_MLP/'
    elif args.model == 'dem':
        folder_path = './saved_model_dem/'
    elif args.model == 'beh':
        folder_path = './saved_model_beh/'
    else:
        folder_path = './saved_model_imsi/'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    save_path = os.path.join(folder_path, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    os.makedirs(save_path)

    cfg = Configuration(vars(args), save_path)
    cfg.configprint()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device
    device = torch.device("cuda")

    torch.autograd.set_detect_anomaly(True)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    writer = SummaryWriter(log_dir=save_path)
    train_file = './dataset_onehot/preprocessed_train_onehot_minmax_dummy.csv'
    val_file = './dataset_onehot/preprocessed_val_onehot_minmax_dummy.csv'
    test_file = './dataset_onehot/preprocessed_test_onehot_minmax_dummy.csv'

    print("cfg.model:",cfg.model)
    if cfg.model == 'BERT_MLP':
        model = BERT_MLPClassification.from_pretrained("klue/bert-base", num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", model_max_length=512)

    elif cfg.model in ['BERT_only', 'dem', 'beh']:
        model = BERT_onlyClassification.from_pretrained("klue/bert-base", num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", model_max_length=512)
    elif cfg.model == 'MLP':
        model = MLP_Classification().to(device)
        tokenizer = None
    else:
        model = BERT_onlyClassification.from_pretrained("klue/bert-base", num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", model_max_length=512)
    # model freeze (only classifier is trained)
    # for name, param in model.bert.named_parameters():
    #     if 'weight' in name:
    #         param.requires_grad = False
    # # add tokens
    # with open('./dataset/add_tokens.pkl', 'rb') as f:
    #     add_tokens = pickle.load(f)
    #
    # if cfg.model != 'MLP':
    #     for token in tqdm(add_tokens, total=len(add_tokens),desc='adding tokens'):
    #         tokenizer.add_tokens(token)
    #     model.resize_token_embeddings(len(tokenizer))

    train_dataset = Baselinedataset(train_file, tokenizer, cfg.model)
    val_dataset = Baselinedataset(val_file, tokenizer, cfg.model)
    test_dataset = Baselinedataset(test_file, tokenizer, cfg.model)

    print("train dataset_onehot : total " + str(train_dataset.__len__()) + 'data')
    print("val dataset_onehot : total " + str(val_dataset.__len__()) + 'data')
    print("test dataset_onehot : total " + str(test_dataset.__len__()) + 'data')

    train_class_counts = np.array(train_dataset.class_counts())
    val_class_counts = np.array(val_dataset.class_counts())
    test_class_counts = np.array(test_dataset.class_counts())
    print("class_counts:", list(train_class_counts + val_class_counts + test_class_counts))
    labels = train_dataset.label
    num_samples = len(train_dataset)


    def create_weights(inp, mode):
        if inp == -1:
            class_counts = train_dataset.class_counts()
            class_weights = [num_samples / class_counts[j] for j in range(len(class_counts))]
            if mode == 'sampling':
                weights = [class_weights[labels[j]] for j in range(int(num_samples))]  # 해당 레이블마다의 가중치 비율
            else:
                weights = class_weights
        elif inp == 0:
            weights = 0
            class_weights = 'Not used'
        else:
            class_weights = [cfg.class_weight, 1 - cfg.class_weight]
            if mode == 'sampling':
                weights = [class_weights[labels[j]] for j in range(int(num_samples))]
            else:
                weights = class_weights

        if mode == 'sampling':
            return torch.DoubleTensor(weights).to(device)
        else:
            return torch.FloatTensor(weights).to(device)

    sampling_weight = create_weights(cfg.sampling_weight, 'sampling')
    class_weight = create_weights(cfg.class_weight, 'class')

    sampler = WeightedRandomSampler(sampling_weight, len(train_dataset), replacement=True)
    if cfg.sampling_weight == 0:
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, drop_last=True, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, drop_last=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    num_warmup_steps = int(train_dataset.__len__() / cfg.batch_size / 20)
    num_training_steps = int(train_dataset.__len__() / cfg.batch_size) * cfg.epochs * 10
    scheduler = linear_lr(optimizer, num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps)
    # scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=3)

    # weight = torch.tensor([cfg.ce_weight]).to(device)
    # if cfg.class_weight != 0:
    #     pos_weight = class_weight[1] / class_weight[0]
    #     print("pos_weight:", pos_weight)
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(cfg.batch_size).to(device)*pos_weight)
    # else:
    #     criterion = nn.BCEWithLogitsLoss()
    if len(class_weight) == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    # criterion = FocalLoss(alpha=1, gamma=0)
    global_steps = 0
    epochs = 0
    max_f1 = 0
    stop_cnt = 0
    min_loss = 100000
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for _ in range(cfg.epochs):
        torch.cuda.empty_cache()
        epochs += 1
        model.train()
        train_loss = 0
        train_acc = 0
        data_cnt = 0
        trn_bunja = 0
        trn_pre_bunmo = 0
        trn_rec_bunmo = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='training'):
            model_input = batch[0].to(device)
            labels = batch[1].to(device)
            input_ids = batch[2].to(device)
            att_mask = batch[3].to(device)
            with torch.cuda.amp.autocast(enabled=False):
                if cfg.model == 'MLP':
                    logits = model(model_input)
                else:
                    logits = model(model_input, input_ids, att_mask)

            # logits = F.softmax(logits, dim=-1)
            # labels_onehot = F.one_hot(labels, num_classes=2)
            loss = criterion(logits, labels)
            train_loss += loss.item()

            # scaler.scale(loss).backward()
            loss.backward()

            if (i + 1) % cfg.acc_steps:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                # scaler.step(scheduler)
                # scaler.update()
                model.zero_grad()
            global_steps += 1
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            acc, precision, recall, f1, bunja, pre_bunmo, rec_bunmo = scoring(logits, labels, 'train')
            trn_bunja += bunja
            trn_pre_bunmo += pre_bunmo
            trn_rec_bunmo += rec_bunmo
            batch_acc = acc / len(logits)
            train_acc += acc
            writer.add_scalar(tag='batch_loss',
                              scalar_value=loss.item(),
                              global_step=global_steps)
            writer.add_scalar(tag='batch_accuracy',
                              scalar_value=batch_acc,
                              global_step=global_steps)
            writer.add_scalar(tag='batch_precision',
                              scalar_value=precision,
                              global_step=global_steps)
            writer.add_scalar(tag='batch_recall',
                              scalar_value=recall,
                              global_step=global_steps)
            writer.add_scalar(tag='batch_f1',
                              scalar_value=f1,
                              global_step=global_steps)
            data_cnt += len(labels)



        train_loss /= data_cnt
        train_acc /= data_cnt
        try:
            trn_pre = trn_bunja / trn_pre_bunmo
        except:
            trn_pre = 0
        try:
            trn_rec = trn_bunja / trn_rec_bunmo
        except:
            trn_rec = 0
        try:
            trn_f1 = 2 * (trn_pre * trn_rec) / (trn_pre + trn_rec)
        except:
            trn_f1 = 0
        writer.add_scalar(tag='train_loss',
                          scalar_value=train_loss,
                          global_step=epochs)
        writer.add_scalar(tag='train_accuracy',
                          scalar_value=train_acc,
                          global_step=epochs)
        writer.add_scalar(tag='train_precision',
                          scalar_value=trn_pre,
                          global_step=epochs)
        writer.add_scalar(tag='train_recall',
                          scalar_value=trn_rec,
                          global_step=epochs)
        writer.add_scalar(tag='train_F1',
                          scalar_value=trn_f1,
                          global_step=epochs)

        model.eval()
        val_loss = 0
        val_acc = 0
        data_cnt = 0
        val_bunja = 0
        val_pre_bunmo = 0
        val_rec_bunmo = 0
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='validation'):
            model_input = batch[0].to(device)
            labels = batch[1].to(device)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    if cfg.model == 'MLP':
                        logits = model(model_input)
                    else:
                        logits = model(model_input, input_ids, att_mask)
            # logits = F.softmax(logits, dim=-1)
            # labels_onehot = F.one_hot(labels, num_classes=2)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().numpy()

            acc, precision, recall, f1, bunja, pre_bunmo, rec_bunmo = scoring(logits, labels, 'val')
            val_bunja += bunja
            val_pre_bunmo += pre_bunmo
            val_rec_bunmo += rec_bunmo
            val_acc += acc
            data_cnt += len(labels)

        val_loss /= data_cnt
        val_acc /= data_cnt
        try:
            val_pre = val_bunja / val_pre_bunmo
        except:
            val_pre = 0
        try:
            val_rec = val_bunja / val_rec_bunmo
        except:
            val_rec = 0
        try:
            val_f1 = 2 * (val_pre * val_rec) / (val_pre + val_rec)
        except:
            val_f1 = 0
        writer.add_scalar(tag='val_loss',
                          scalar_value=val_loss,
                          global_step=epochs)
        writer.add_scalar(tag='val_accuracy',
                          scalar_value=val_acc,
                          global_step=epochs)
        writer.add_scalar(tag='val_precision',
                          scalar_value=val_pre,
                          global_step=epochs)
        writer.add_scalar(tag='val_recall',
                          scalar_value=val_rec,
                          global_step=epochs)
        writer.add_scalar(tag='val_F1',
                          scalar_value=val_f1,
                          global_step=epochs)

        print("=============================")
        print("epoch", epochs)
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("train_precision:", trn_pre)
        print("train_recall:", trn_rec)
        print("train_f1:", trn_f1)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)
        print("val_precision:", val_pre)
        print("val_recall:", val_rec)
        print("val_f1:", val_f1)

        # f1 기반 early stopping
        if val_f1 > max_f1:
            max_f1 = val_f1
            stop_cnt = 0
            is_best = True
        else:
            stop_cnt += 1
            is_best = False

        # val_loss 기반 early stopping
        # if val_loss < min_loss:
        #     min_loss = val_loss
        #     stop_cnt = 0
        #     is_best = True
        # else:
        #     stop_cnt += 1
        #     is_best = False

        if is_best:
            state = {
                'epoch': epochs,
                'model': model,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'precision': val_pre,
                'recall': val_rec,
                'f1-score': val_f1,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, save_path + '/model_best.pth.tar')

        if stop_cnt > cfg.threshold:
            print("Training finished.")
            break

    # evaluation
    model_state_dict = torch.load(save_path + '/model_best.pth.tar', map_location=device)['state_dict']
    model.load_state_dict(model_state_dict)

    model.eval()
    eval_acc = 0
    data_cnt = 0
    eval_bunja = 0
    eval_pre_bunmo = 0
    eval_rec_bunmo = 0
    for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='testing:'):
        model_input = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                if cfg.model == 'MLP':
                    logits = model(model_input)
                else:
                    logits = model(model_input, input_ids, att_mask)
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
    with open(save_path + '/test_result.txt', 'w') as f:

        f.write("eval_precision: " + str(eval_pre) + '\n')
        f.write("eval_recall: " + str(eval_rec) + '\n')
        f.write("eval_F1 : " + str(eval_f1) + '\n')
        f.write("eval_acc: " + str(eval_acc) + '\n')
