from __future__ import absolute_import

import os, re, collections
import requests, nltk

import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from TF2.extract_features_Builtin import *

type = 'bert'
if type == 'bert':
    bert_folder = 'Pretrained/uncased_L-12_H-768_A-12/'

    bert_config = bert_folder + 'bert_config.json'
    vocab_file  = bert_folder + 'vocab.txt'
    bert_ckpt   = bert_folder + 'bert_model.ckpt'

pmc_id = '4304705'
url = 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:'+pmc_id+'&metadataPrefix=pmc'
d = requests.get(url).content.decode('utf-8')
xmldata = re.sub('xmlns="[^"]+"', '', d)
xml_handle = ET.fromstring(xmldata)

# get abstact sentences from xml
abstract = xml_handle.findall('.//abstract')
abs_text = ET.tostring(abstract[0],method='text').decode('utf-8')
abs_text = re.sub('\n',' ',abs_text)
abs_text = re.sub(r'\s+',' ',abs_text)
abs_sents = nltk.sent_tokenize(abs_text)

tf.compat.v1.logging.set_verbosity('ERROR')
# Return vectors in pandas frame
Emb_Vectors = Ext_Features(input=abs_sents, bert_config_file=bert_config, vocab_file=vocab_file, init_checkpoint=bert_ckpt,
             input_type='string', layers = '-1', max_seq_length=128, do_lower_case=True, batch_size=32,
             use_tpu = False, master = None, num_tpu_cores=8, use_one_hot_embeddings=False)

Emb_Vectors.head(5)
