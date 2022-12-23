import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    add_tokens = []
    add_tokens.append('정보없음')
    with open('./dataset/dict_제1전공.pickle', 'rb') as f:
        major = pickle.load(f)
        major = set(major.values())
        for this_major in major:
            add_tokens.append(this_major)

    with open('./dataset/dict_학부과.pickle', 'rb') as f:
        depart = pickle.load(f)
        depart = set(depart.values())
        for this_depart in depart:
            add_tokens.append(this_depart)

    train_df = pd.read_csv('./dataset/preprocessed_train_embed_all_2022-09-18_1학년1학기.csv', encoding='cp949')
    high_school = set(train_df['출신고교'].tolist())
    for this_high in high_school:
        add_tokens.append(this_high)

    ent = set(train_df['입학구분'].tolist())
    for this_ent in ent:
        add_tokens.append(this_ent)

    scr = set(train_df['전형구분'].tolist())
    for this_scr in scr:
        add_tokens.append(this_scr)



    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", model_max_length=512)

    new_tokens = []
    for token in tqdm(add_tokens, total=len(add_tokens)):
        if token not in tokenizer.vocab.keys():
            new_tokens.append(token)

    new_tokens = set(new_tokens)
    with open('./dataset/add_tokens.pkl', 'wb') as f:
        pickle.dump(new_tokens, f)

