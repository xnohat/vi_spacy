#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install pyvi


# In[2]:


#pip install plac


# ## Load Packages

# In[3]:


#get_ipython().system('python -m spacy download en')


# In[4]:


from __future__ import unicode_literals, print_function
import plac
import pickle
import random
from pathlib import Path
import spacy
from tqdm import tqdm 


# In[5]:


nlp1 = spacy.load('/root/miniconda3/envs/spacyner/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.1.0')


# ## Working of NER

# In[6]:


docx1 = nlp1(u"Who is Nishanth?")


# In[7]:


for token in docx1.ents:
    print(token.text,token.start_char, token.end_char,token.label_)


# In[8]:


docx2 = nlp1(u"Who is Kamal Khumar?")


# In[9]:


for token in docx2.ents:
    print(token.text,token.start_char, token.end_char,token.label_)


# ## Train Data

# In[10]:


with open ('./phoner_train_word_spacy_ner', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)


# ## Define our variables

# In[11]:


model = None
output_dir=Path("./ner")
n_iter=100


# ## Load the model

# In[12]:


if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('vi')  
    print("Created blank 'vi' model")


# ## Set up the pipeline

# In[13]:


if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')


# ## Train the Recognizer

# In[ ]:


for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],  
                [annotations],  
                drop=0.5,  
                sgd=optimizer,
                losses=losses)
        print(losses)


# ## Test the trained model

# In[ ]:


for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# ## Save the model

# In[ ]:


if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)        


# ## Test the saved model

# In[ ]:


print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# In[ ]:




