# This Python 3 environment comes with many helpful analytics libraries installed
#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Libraries for text

# !pip install PyDrive
# !pip install gensim
# !pip install pyldavis
# !python -m spacy download en
from nltk.corpus import stopwords
from nltk.util import ngrams

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from nltk.tokenize import TweetTokenizer
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score

#!pip install wordcloud
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


from bs4 import BeautifulSoup
import requests as rq
import urllib.request as url
from bs4 import BeautifulSoup as bs

import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import re
import spacy
import gzip
import en_core_web_sm

import gensim
from gensim import corpora

import os
import warnings;
warnings.filterwarnings("ignore");


# Visulization
import plotly
import plotly.offline as pyoff

import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# !pip install vecstack
# !pip -q install shap
# !pip -q install lime
# !pip -q install eli5
# !pip install tpot
# !pip install hyperopt
#!pip install xgboost
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import random as rnd
pd.set_option('max_colwidth',400)
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Importing Models
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

# Importing other tools
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV



%%time

all_links['link_text'] = ""

def find_between(s,start,end):
    return (s.split(start))[1].split(end)[0]

start = '<TYPE>10-K'
end = '</DOCUMENT>'
result=[]

for rownum, row in stock.iterrows():
    try:
        html = row['link']
        link = url.urlopen(html).read()
        link_data = link.decode('utf-8')
        result = find_between(link_data,start,end)
        soup = bs(result,'lxml')
        text =soup.find_all(text=True)
        blacklist = ['a','sequence','filename','description']
        output =""

    for t in text:
        
          if t.parent.name not in blacklist:
            output += '{}'.format(t)
      # print(output)
      
    
    except:
        pass
all_links['link_text'][rownum]=output