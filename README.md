# STUD-AI
This is PyTorch implementation of Academic Information-based Early Warning System for University Dropout.


## Environment
The code requires the CUDA11.6 toolkit.

**Install basic dependencies**

pip install -r requirements.txt



## Datasets
Dataset is currently not available.
(Dataset cotains personal information of students)

## Configuration
hyperparameters of the framework (config.py)  

--lr : learning rate for training  
--batch_size : batch size for training  
--device : your gpu number  
--threshold : early-stopping threshold  
--sampling_weight : setting sampling weight for data imbalance. 0 is not use, -1 for real sample ratio  
--class_weight : setting different class weight for data imbalance. 0 is not use, -1 for real sample ratio  
--maxlen : maximum length of the sentences  
--model : model name for training and prediction. MLP / BERT_MLP / BERT_only / RoBERTa_only / dem / beh  
    * dem : use only demographic information, beh : use only behavioral information  
--acc_steps : gradient accumulation steps  
--epochs : training epochs  



## Training and Prediction

### Fine-tuning
python main.py --model dem --lr 5e-5 --acc_steps 128 --threshold 7 --class_weight 0 --batch_size 16 --epochs 15

