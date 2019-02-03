# Quora Insincere Question Classification
 Kaggle competition to classify insincere questions asked in quora using machine learning models.
 
### Python dependencies necessary to import:

```
from google.colab import drive
import pandas as pd
import zipfile
import numpy as np 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout
from keras import backend as K

```



### Dataset: https://www.kaggle.com/c/quora-insincere-questions-classification/data

### Dataset Description
  Train and test data are stored in two separate csv files in text format.
  ![Screenshot](snapshot.png)
  We load them in pandas dataframe.
  ```
  trainDf=pd.read_csv("train.csv")
  ```
  
##### Train Data 
1.31 million questions are labelled either as insincere or not by 1 and 0 respectively. Each question
  has a unique id. Each row consists of three columns: qid, question_text, target.
  
##### Test Data
As test data provided in kaggle is not labelled, we split the training data into training, test and validation 
sets by 60, 20 and 20 percents using train_test_split from scikit learn.
```
trainDf, valDf = train_test_split(trainDf, test_size=0.4)
valDf, testDf = train_test_split(valDf, test_size=0.5)
```

###
  
