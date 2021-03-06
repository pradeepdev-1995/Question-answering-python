# -*- coding: utf-8 -*-
"""QA_ktrain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/131rwu63JQAW60lNkOQY_UO4AxuJRkp5B
"""

!pip3 install -q ktrain

from sklearn.datasets import fetch_20newsgroups
remove = ('headers', 'footers', 'quotes')
newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
docs = newsgroups_train.data +  newsgroups_test.data

import ktrain
from ktrain import text

INDEXDIR = '/tmp/myindex'

text.SimpleQA.initialize_index(INDEXDIR)
text.SimpleQA.index_from_list(docs, INDEXDIR, commit_every=len(docs))

qa = text.SimpleQA(INDEXDIR)

answers = qa.ask('When did the Cassini probe launch?')
qa.display_answers(answers[:5])

answers = qa.ask('What causes computer images to be too dark?')
qa.display_answers(answers[:5])

answers = qa.ask('Who was Jesus Christ?')
qa.display_answers(answers[:5])

answers = qa.ask('Who is sachin tendulkkar?')
qa.display_answers(answers[:5])

answers = qa.ask('What is solar panel battery')
qa.display_answers(answers[:5])