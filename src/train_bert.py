import json
import os
import sys
from torch.utils.data import Dataset, DataLoader
from data import PropaDataset

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig
from modeling_bert import BertForSequenceClassificationAddSymbolicFeatures
from transformers import AdamW
import torch
from torch import nn
import random
import numpy as np
from tqdm import tqdm
from scorer_task1_3 import evaluate_pred_file
from format_checker_task1_3 import read_classes


def train(train_dataloader, model, loss_fct, optimizer, scheduler, device, config):
    model.to(device)
    model.train()
    
    total_train_loss = 0
    last_batch_loss = 0
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        if step % 2 == 0 and not step == 0:
            print(f"{step}/{len(train_dataloader)} | last batch loss:{last_batch_loss}")

        if not config['add_symbolic_features']:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
        else:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_symbolic_features = batch[3].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, symbolic_features = b_symbolic_features, labels=b_labels)
            logits = outputs.logits

        # loss = outputs.loss
        loss = loss_fct(logits, b_labels)

        total_train_loss += loss.item()
        last_batch_loss = loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    print("avg_train_loss:",avg_train_loss)
    return model

def evaluate(data_loader, model, loss_fct, device, config, phase):    
    model.to(device)
    model.eval()
    
    predictions = []
    total_eval_loss = 0
    # For each batch of training data...

    for step, batch in enumerate(data_loader):

        if not config['add_symbolic_features']:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_index = batch[3]

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
        else:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_symbolic_features = batch[3].to(device)
            b_index = batch[4]

            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask, symbolic_features = b_symbolic_features, labels=b_labels)
            logits = outputs.logits

        loss = loss_fct(logits, b_labels)
        total_eval_loss += loss.item()

        pred_soft = torch.sigmoid(logits[0]).detach().cpu().numpy()
        pred = np.where(pred_soft >= config['threshold'], 1, 0)
        # print(pred)
        pred_prop_indices = np.where(pred==1)[0]
        # print(pred_prop_indices)
        index = int(b_index[0].detach().cpu().numpy())
        pred_ann_item = data_loader.dataset.ann[index].copy()
        # print(pred_ann_item)
        pred_ann_item['labels'] = [data_loader.dataset.index_to_prop[prop_idx] for prop_idx in pred_prop_indices]
        # print(pred_ann_item)
        predictions.append(pred_ann_item)
    
    # output prediction
    if config['evaluate']:
        print('write out prediction..')
        threshold = config['threshold']
        output_prediction_path = os.path.join(config['output_dir'],f'predictions_{phase}-threshold_{threshold}.txt') 
        with open(output_prediction_path,'w') as out:
            json.dump(predictions, out, indent=4)
    else:
        output_prediction_path = None
    
    avg_eval_loss = total_eval_loss / len(data_loader)
    print("avg_eval_loss:",avg_eval_loss)
    return avg_eval_loss, output_prediction_path

def get_pos_weights(train_dataset):
    negative = np.array([0 for i in range(len(train_dataset.prop_to_index))])
    positive = np.array([0 for i in range(len(train_dataset.prop_to_index))])
    for item in train_dataset.ann:
        negative += 1 # all add negative 1
        for prop in item['labels']:
            idx = train_dataset.prop_to_index[prop]
            negative[idx] -= 1 # reverse adding negative
            positive[idx] += 1 # adding positive
    positive += 1 # smoothing
    pos_weights_train = np.array([negative[i]/positive[i] for i in range(len(positive))])
    print('using pos_weight:')
    print(pos_weights_train)
    return torch.from_numpy(pos_weights_train)


def main(config):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # set device  
    device = torch.device(config['device'])
    
    # paths
    ann_path_train = "../SEMEVAL-2021-task6-corpus/data/training_set_task1.txt"
    ann_path_dev = "../SEMEVAL-2021-task6-corpus/data/dev_set_task1.txt"
    ann_path_test = "../SEMEVAL-2021-task6-corpus/data/test_set_task1.txt"
    techniques_list_task1_path = '../SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt'

    # set up tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # datasets
    train_dataset = PropaDataset(ann_path_train, techniques_list_task1_path, tokenizer, max_length = 128, add_symbolic_features = config['add_symbolic_features'])
    dev_dataset = PropaDataset(ann_path_dev, techniques_list_task1_path, tokenizer, max_length = 128, add_symbolic_features = config['add_symbolic_features'])
    test_dataset = PropaDataset(ann_path_test, techniques_list_task1_path, tokenizer, max_length = 128, add_symbolic_features = config['add_symbolic_features'])
    print('train set size:',len(train_dataset))
    print('dev set size:',len(dev_dataset))
    print('test set size:',len(test_dataset))
    if config['add_pos_weight']:
        pos_weights_train = get_pos_weights(train_dataset).to(device)
    else:
        pos_weights_train = torch.ones([len(train_dataset.prop_to_index)]).to(device)
    # dataloaders
    batch_size = config['batch_size']
    if not config['evaluate']:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # set up pretrained model
    if config['add_symbolic_features']:
        num_symbolic_features = train_dataset[0][3].shape[0]
        print(f'adding {num_symbolic_features} symbolic features')
        pretrain_config = BertConfig.get_config_dict("bert-base-uncased")[0]
        pretrain_config['num_symbolic_features'] = num_symbolic_features
        pretrain_config['num_labels'] = len(train_dataset.prop_to_index)
        print('using config: ', pretrain_config)
        print('==================================')
        configuration = BertConfig.from_dict(pretrain_config)
        configuration.update(pretrain_config)

        model = BertForSequenceClassificationAddSymbolicFeatures.from_pretrained("bert-base-uncased", config = configuration)
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification",num_labels=len(train_dataset.prop_to_index))
    
    # optimizer and schedular
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_train)

    # setup schedular
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_dataloader) * config['epoch']
    print("total_steps:" , total_steps)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 10, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    min_loss = 100000000
    for epoch in range(config['epoch']):
        print(f'== epoch {epoch} ==')
        if not config['evaluate']:
            # train
            model = train(train_dataloader, model, loss_fct, optimizer, scheduler, device, config)
        else:
            if config['checkpoint']:
                print('loading checkpoint:',config['checkpoint'])
                model.load_state_dict(torch.load(config['checkpoint']))
        
        # eval dev
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"])

        dev_loss, prediction_path = evaluate(dev_dataloader, model, loss_fct, device, config, phase = 'dev')
        if dev_loss < min_loss and (not config['evaluate']):
            min_loss = dev_loss
            print("saving best checkpoint...")
            torch.save(model.state_dict(), os.path.join(config["output_dir"],f'checkpoint_best.pt'))
        # score prediction:
        if prediction_path:
            dev_macro_f1, dev_micro_f1 = evaluate_pred_file(prediction_path,ann_path_dev,dev_dataset.prop_to_index.keys())
            print(f'dev score:\n\tmacro_f1:{dev_macro_f1}, micro_f1:{dev_micro_f1}')
            
            # search on train set instead of dev set
            # train_loss, prediction_path = evaluate(train_dataloader, model, loss_fct, device, config, phase = 'train')
            # dev_macro_f1, dev_micro_f1 = evaluate_pred_file(prediction_path,ann_path_train,train_dataset.prop_to_index.keys())
            # print(f'dev score:\n\tmacro_f1:{dev_macro_f1}, micro_f1:{dev_micro_f1}')
        
        # eval test
        test_loss, prediction_path = evaluate(test_dataloader, model, loss_fct, device, config, phase = 'test')
        if prediction_path:
            test_macro_f1, test_micro_f1 = evaluate_pred_file(prediction_path,ann_path_test,test_dataset.prop_to_index.keys())
            print(f'test score:\n\tmacro_f1:{test_macro_f1}, micro_f1:{test_micro_f1}')

        if config['evaluate']:
            break
    
    if prediction_path:
        return dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1
    else:
        return None,None,None,None

def random_baseline_score():
    ann_path_train = "../SEMEVAL-2021-task6-corpus/data/training_set_task1.txt"
    ann_path_dev = "../SEMEVAL-2021-task6-corpus/data/dev_set_task1.txt"
    ann_path_test = "../SEMEVAL-2021-task6-corpus/data/test_set_task1.txt"
    techniques_list_task1_path = '../SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt'
    CLASSES = read_classes(techniques_list_task1_path)
    
    dev_prediction_path = '../SEMEVAL-2021-task6-corpus/baselines/baseline-output-task1-random-dev.txt'
    test_prediction_path = '../SEMEVAL-2021-task6-corpus/baselines/baseline-output-task1-random-test.txt'
    
    dev_macro_f1, dev_micro_f1 = evaluate_pred_file(dev_prediction_path,ann_path_dev,CLASSES)
    print(f'dev score:\n\tmacro_f1:{dev_macro_f1}, micro_f1:{dev_micro_f1}')
    test_macro_f1, test_micro_f1 = evaluate_pred_file(test_prediction_path,ann_path_test,CLASSES)
    print(f'test score:\n\tmacro_f1:{test_macro_f1}, micro_f1:{test_micro_f1}')



if __name__ == '__main__':
    '''run bert'''
    phase = 'train' # train
    # phase = 'search_threshold' # perform hyper-parameter search on the best threshold on dev set
    if phase == 'train':
        config = {
            "add_symbolic_features":True,
            "evaluate":False,
            "batch_size":32,
            "device":'cuda',
            "output_dir":"../output/train/b_32_add_symbolic",
            "checkpoint":None,
            "lr":0.01,
            "epoch":10,
            "add_pos_weight":False,
            "threshold":0.5
        }
        dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1 = main(config)
    elif phase == 'search_threshold':
        # search threshold on dev:
        best_t = -1
        best_dev_avg_f1 = 0
        for t in np.linspace(0,1,num=20)[1:-1]:
            if t>0.5:
                break
            print(f'running threshold:{t}')
            config = {
                "add_symbolic_features":True,
                "evaluate":True,
                "batch_size":32,
                "device":'cuda:2',
                "output_dir":"../output/evaluate/b_32_add_symbolic",
                "checkpoint":'../output/train/b_32_add_symbolic/checkpoint_best.pt',
                "lr":0.01,
                "epoch":10,
                "add_pos_weight":False,
                "threshold":t
            }
            dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1 = main(config)
            if (dev_macro_f1+dev_micro_f1)/2 > best_dev_avg_f1:
                best_dev_avg_f1 = (dev_macro_f1+dev_micro_f1)/2
                best_t = t
        print('best threshold:',best_t)

    '''evaluation'''
    # checkpoint = '<checkpoint_path>'
    # t = <best_threshold>
    # print(f'running threshold:{t}')
    # config = {
    #     "add_symbolic_features":True,
    #     "evaluate":True,
    #     "batch_size":32,
    #     "device":'cuda:2',
    #     "output_dir":"../output/evaluate/b_32_add_symbolic_all",
    #     "checkpoint":checkpoint,
    #     "lr":0.01,
    #     "epoch":10,
    #     "add_pos_weight":False,
    #     "threshold":t
    # }
    # dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1 = main(config)
    # print('dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1:', dev_macro_f1, dev_micro_f1, test_macro_f1, test_micro_f1)


    '''run random baseline'''
    # random_baseline_score()
    
    