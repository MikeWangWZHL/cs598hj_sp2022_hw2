import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import json
from transformers import BertTokenizer, BertModel
import spacy

def load_propaganda_techniques_file(propaganda_techniques_file):
    with open(propaganda_techniques_file, "r") as f:
        propaganda_techniques_names = [ line.rstrip() for line in f.readlines() if len(line)>2 ]
    # propaganda_techniques_names.insert(0,'UNK')
    prop_to_index = {propaganda_techniques_names[i]:i for i in range(len(propaganda_techniques_names))}
    index_to_prop = {value:key for key,value in prop_to_index.items()}
    return prop_to_index, index_to_prop

def load_annotation(json_file):
    try:
        with open(json_file, "r") as f:
            jsonobj = json.load(f)
    except:
        sys.exit("ERROR: cannot load json file")
    return jsonobj

class PropaDataset(Dataset):
    """Propaganda dataset."""
    def __init__(self, json_file, propaganda_techniques_file, tokenizer, max_length = 128, image_dir = None, transform=None, add_symbolic_features = False):
        self.prop_to_index, self.index_to_prop = load_propaganda_techniques_file(propaganda_techniques_file)
        print(self.prop_to_index)
        self.ann = load_annotation(json_file)

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.add_symbolic_features = add_symbolic_features  

        # spacy
        self.nlp = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        item = self.ann[idx]

        labels = [0 for i in range(len(self.prop_to_index))]
        for prop in item['labels']:
            prop_idx = self.prop_to_index[prop]
            labels[prop_idx] = 1
        labels = torch.tensor(labels, dtype=torch.float)

        text = item['text']
        tokenized = self.tokenizer(text,return_tensors = 'pt', max_length = self.max_length, padding = 'max_length', truncation = True)
        input_ids = tokenized['input_ids'][0]
        attention_masks = tokenized['attention_mask'][0]

        if self.add_symbolic_features:
            # symbolic features
            symbolic_features = self._get_symbolic_features(text)
            symbolic_features = torch.tensor(list(symbolic_features), dtype=torch.float)
            return input_ids, attention_masks, labels, symbolic_features, idx
        else:
            return input_ids, attention_masks, labels, idx

    def _get_symbolic_features(self, text):
        doc = self.nlp(text)
        text_length = len(doc)
        non_stopwords = set()

        quotation_mark_count = 0
        question_mark_count = 0
        exclamation_mark_count = 0
        repeated_non_stopword_count = 0
        adv_count = 0
        adj_count = 0
        
        for token in doc:
            # quotation
            if token.text in ['\"',"``","\'"]:
                quotation_mark_count += 1
            elif token.text == "?":
                question_mark_count += 1
            elif token.text == "!":
                exclamation_mark_count += 1

            # repeatition
            if not token.is_stop and not (token.pos_ == 'PUNCT') and not "\n":
                if token.text in non_stopwords:
                    repeated_non_stopword_count += 1
                non_stopwords.add(token.text)

            # adv, adj
            if token.pos_ == 'ADJ':
                adj_count += 1
            elif token.pos_ == 'ADV':
                adv_count += 1
        
        named_entity_count = len(doc.ents)
        gpe_count = 0
        for ent in doc.ents:
            if ent.label_ == "GPE":
                gpe_count += 1

        # normalize:
        quotation_mark_count /= text_length
        question_mark_count /= text_length
        exclamation_mark_count /= text_length
        repeated_non_stopword_count /= text_length
        adv_count /= text_length
        adj_count /= text_length
        named_entity_count /= text_length
        gpe_count /= text_length

        return (
            quotation_mark_count,
            question_mark_count,
            exclamation_mark_count,
            repeated_non_stopword_count,
            adv_count,
            adj_count,
            named_entity_count,
            gpe_count
        )
        # return (
        #     quotation_mark_count,
        #     question_mark_count,
        #     exclamation_mark_count,
        #     named_entity_count,
        #     gpe_count
        # )


if __name__ == '__main__':
    json_file = '/shared/nas/data/m1/wangz3/cs598hj_sp2022/assignment2/SEMEVAL-2021-task6-corpus/data/dev_set_task1.txt'
    propaganda_techniques_file = '/shared/nas/data/m1/wangz3/cs598hj_sp2022/assignment2/SEMEVAL-2021-task6-corpus/techniques_list_task1-2.txt'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
    
    dataset = PropaDataset(json_file, propaganda_techniques_file, tokenizer)
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])