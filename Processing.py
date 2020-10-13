import os
import random
import json

import numpy as np 
import torch

from transformers import BertTokenizer, BertForTokenClassification, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm
import time
import datetime
import logging
import yaml
import logging.config
#%%
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

RANDOM_STATE = 24
# reprobducible
random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
###################################################################################################
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

from pyltp import SentenceSplitter
from pyltp import Segmentor
LTP_DATA_DIR = 'E:/LTP_Model/ltp_data_v3.4.0'    # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
#%%
# Define some global variables
types = ['none', 'B-time', 'I-time', 'B-location', 'I-location', 'B-pn', 'I-pn', 'B-on', 'I-on', 'B-cn', 'I-cn', 'B-pn', 'I-pn', 'cls_token', 'sep_token', 'pad_token']
entity_types = {'none':0, 'time': 1, 'location': 3, 'person_name': 5, 'org_name': 7, 'company_name': 9, 'product_name': 11, 'cls_token': 13, 'sep_token': 14, 'pad_token': 15}
token_dict = {'unk_token':'[UNK]', 'sep_token':'[SEP]', 'pad_token':'[PAD]', 'cls_token':'[CLS]', 'mask_token':'[MASK]', 'padding_side':'right'}
MAX_LENGTH = 80
NUM_CLASSES = 16    # the losses come from classifying pad_token will not be included in the final loss 
#%%

    

class TracksAnalysis:
    def __init__(self):
        #self.segmentor = Segmentor()
        #self.segmentor.load(cws_model_path)
        self.model, self.tokenizer = self._load_fine_tuned_model()
        self.model.to(device)
        self.model.eval()   # 只使用，不训练
           
    def _text_to_sentences(self, text):
        '''
        使用 pyltp 对文本分句
        '''
        if not text: return None
        return list(SentenceSplitter.split(text))
    
    def _load_json_data(self, data_dir='patient_tracks'):
        with open(data_dir + '.json', 'r') as f:
            data = json.load(f)
            print(type(data))
            return data
        return None

    def _load_fine_tuned_model(self, model_dir='../pre_training_models/BERT_Chinese/model_save'):
        '''
        Load a trained model and vocabulary that we have fine-tuned from the directory "model_dir"
        '''
        model = BertForTokenClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        logging.info("Load fine-tuned model and tokenizer from model directory successfully!")
        return model, tokenizer

    def _model_process(self, words):
        '''
        words 不需要经过分词，为单个句子。
        使用 BertForTokenClassification 从 words 中抽取出所需的命名实体（地名、机构名、公司名）
        return: a list of tuples [(word_type, word),...]
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(
            words,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True,
            return_attention_masks=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**encoded_dict)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        input_ids = np.array(encoded_dict['input_ids'])
        input_masks = np.array(encoded_dict['attention_mask'])
        preds = preds * input_masks

        named_entity_index = np.argwhere((preds==3)|(preds==4)|(preds==7)|(preds==8)|(preds==9)|(preds==10))
        loc_words_index = np.argwhere((preds==3)|(preds==4))
        org_name_index = np.argwhere((preds==7)|(preds==8))
        comp_name_index = np.argwhere((preds==9)|(preds==10))
        
        if named_entity_index.size == 0: return []
        words = [] # list of tuples: [(word_type, word),..]
        word = ''
        previous = named_entity_index[0,0] * MAX_LENGTH + named_entity_index[0,1] - 1
        for i, index in enumerate(named_entity_index):
            assert len(index) == 2
            
            row = index[0]
            col = index[1]
            ids = input_ids[row, col]
            token = self.tokenizer.convert_ids_to_tokens(int(ids))
            word += token

            # 如果前后两个索引对应的字在原始句子中的位置不是邻近的，
            # 那么就要保存当前字之前的字组合成的词
            if row * MAX_LENGTH + col != previous + 1:  

                if len(word[:-1]) > 1:  # 过滤掉字数少于 2 的实体
                    # 判断该实体的类型
                    word_type = ""
                    if named_entity_index[i-1] in loc_words_index:
                        word_type = "location"
                    elif named_entity_index[i-1] in org_name_index:
                        word_type = "organization"
                    elif named_entity_index[i-1] in comp_name_index:
                        word_type = "company"

                    words.append((word_type, word[:-1]))
                    print(word_type, word[:-1])

                word = token
            # 保存索引在最后面的字
            if i == len(named_entity_index)-1: 

                # 判断该实体的类型
                word_type = ""
                if named_entity_index[i] in loc_words_index:
                    word_type = "location"
                elif named_entity_index[i] in org_name_index:
                    word_type = "organization"
                elif named_entity_index[i] in comp_name_index:
                    word_type = "company"

                words.append((word_type, word))

            previous = row * MAX_LENGTH + col
        
        return words

    def _index_to_words(self, indcies, word_ids):
        '''
        根据 indcies 从 word_ids 中抽取对应索引位置上的字索引，然后使用字索引从字典中对应的字。
        '''
        if indcies[0].size == 0: return None
        rows, cols = indcies
        words = []
        word = ''
        previous = rows[0] * MAX_LENGTH + cols[0] - 1
        for i, row in enumerate(rows):
            col = cols[i]
            ids = word_ids[row, col]
            token = self.tokenizer.convert_ids_to_tokens(int(ids))
            word += token

            if row * MAX_LENGTH + col != previous + 1:
                words.append(word[:-1])
                word = token

            if i == len(rows)-1: words.append(word)

            previous = row * MAX_LENGTH + col

        return words

    def process(self):
        # Load data
        track_list = self._load_json_data()
        span_records = []
        # Read data
        for i, per_page in enumerate(tqdm(track_list)):
            report_date = per_page['report_date']
            city = per_page['city']
            patient_list = per_page['patient_list']

            for j, per_patient in enumerate(patient_list):
                patient_id = per_patient['id']
                patient_address = per_patient['address']
                patient_tracks = per_patient['tracks']

                tracks_processed = [] #[{track_date: named_entites}]
                time_span = ''
                
                if len(patient_tracks) == 0:
                    # 只有地址没有轨迹
                    # 添加 tracks_processed 属性
                    per_patient['tracks_processed'] = tracks_processed
                    # 添加 time_span 属性
                    per_patient['time_span'] = time_span
                    continue
                first_date = patient_tracks[0][0]   # 轨迹记录的起始日期
                for k, per_track in enumerate(patient_tracks):
                    track_date = per_track[0]   # 该条轨迹对应的日期
                    track_record = per_track[1] # 该条轨迹的文本数据
                    # 对轨迹文本分句
                    words = self._text_to_sentences(track_record)
                    # 获取该条轨迹中包含的位置名，机构名，公司名
                    named_entites = self._model_process(words)  # a list of tuples
                    tracks_processed.append({track_date: named_entites})  
                    last_date = track_date if track_date else last_date # 轨迹记录的终止日期

                # 检查轨迹日期是否合理
                first_date = datetime.date(*[int(i) for i in first_date.split('-')])
                last_date = datetime.date(*[int(i) for i in last_date.split('-')])
                assert first_date <= last_date
                # 轨迹记录的时间跨度
                time_span = (last_date-first_date).days
                # 添加 tracks_processed 属性
                per_patient['tracks_processed'] = tracks_processed
                # 添加 time_span 属性
                per_patient['time_span'] = time_span
                # 统计
                span_records.append(int(time_span))
                print(first_date, last_date)

        print("平均跨度时间：", sum(span_records) / len(span_records))
        with open('patient_tracks_processed.json', 'w+') as f:
            json.dump(track_list, f)
        return 'patient_tracks_processed'
   
if __name__ == '__main__':  


    
    #load_language_tools() 
    analysis = TracksAnalysis()
    file_name = analysis.process()
