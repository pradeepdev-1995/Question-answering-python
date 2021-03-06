# -*- coding: utf-8 -*-
"""QA_haystack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Jwy5dR9yV800OUc4UEFnFlT9QrcwdjI
"""

!pip3 install farm-haystack

!pip3 install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

"""Question answering with Elastic search"""

! wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz -q

! tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz

! chown -R daemon:daemon elasticsearch-7.6.2

import os
from subprocess import Popen, PIPE, STDOUT
es_server = Popen(['elasticsearch-7.6.2/bin/elasticsearch'],
                   stdout=PIPE, stderr=STDOUT,
                   preexec_fn=lambda: os.setuid(1)  # as daemon
                  )
# wait until ES has started
! sleep 30

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document")

doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)


print(dicts[:3])

document_store.write_documents(dicts)

from haystack.retriever.sparse import ElasticsearchRetriever
retriever = ElasticsearchRetriever(document_store=document_store)



reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

finder = Finder(reader, retriever)

prediction = finder.get_answers(question="Who is the education minister", top_k_retriever=10, top_k_reader=5)

print_answers(prediction, details="minimal")

"""Question answering without Elastic search"""

from haystack.document_store.memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)


print(dicts[:3])
document_store.write_documents(dicts)

from haystack.retriever.sparse import TfidfRetriever
retriever = TfidfRetriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

finder = Finder(reader, retriever)

prediction = finder.get_answers(question="Who is the father of computer?", top_k_retriever=10, top_k_reader=5)

print_answers(prediction, details="minimal")