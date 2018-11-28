
# coding: utf-8

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import FactorAnalysis
#Dressing The Data Set
df = pd.read_csv('Most-Recent-Cohorts-Treasury-Elements.csv')

def BestModel(target):
    tg = target
    column_selector=[] # picks columns 5 thru 12 and 14 through the rest
    for i in range(92):
            if(i<5 or i==tg): #these columns are exclude from the dataset
                column_selector.append(False)
            else:
                column_selector.append(True) #these are accepted

    
    PcaX = df.iloc[:, column_selector ].values   # features
    Pcay = df.iloc[:, tg].values #target MEDIAN HH INC


    LdaX = df.iloc[:, column_selector].values   
    Lday = df.iloc[:, tg].values

    FaX = df.iloc[:, column_selector].values  
    Fay = df.iloc[:, tg].values

    PcaX_train, PcaX_test, Pcay_train, Pcay_test = train_test_split(PcaX, Pcay, test_size=0.25)
    LdaX_train, LdaX_test, Lday_train, Lday_test = train_test_split(LdaX, Lday, test_size=0.25)
    FaX_train, FaX_test, Fay_train, Fay_test = train_test_split(FaX, Fay, test_size=0.25)
    
    print(df.columns[tg], " Is your Target")


# In[2]:


#Principal Component Analysis
# Always scale data for good results on PCA

    PcaX_scale = StandardScaler()
    PcaX_train = PcaX_scale.fit_transform(PcaX_train)
    PcaX_test = PcaX_scale.transform(PcaX_test)

#PCA Part

    pca = PCA(n_components=None)
    pca.fit_transform(PcaX_train)
    pca = PCA(n_components=2)
    PcaX_train = pca.fit_transform(PcaX_train)
    PcaX_test = pca.transform(PcaX_test)


# In[3]:



# Split in training and testing

# Scale
    LdaX_scale = StandardScaler()
    LdaX_train = LdaX_scale.fit_transform(LdaX_train)
    LdaX_test = LdaX_scale.transform(LdaX_test)


    lda = LinearDiscriminantAnalysis(n_components=2)
# since LDA is supervised, also need to pass y_train
    LdaX_train = lda.fit_transform(LdaX_train.astype(int), Lday_train.astype(int))
    LdaX_test = lda.transform(LdaX_test)


# In[4]:



# Split in training and testing


# Always scale data for good results on PCA
    FaX_scale = StandardScaler()
    FaX_train = FaX_scale.fit_transform(FaX_train)
    FaX_test = FaX_scale.transform(FaX_test)


    fa = FactorAnalysis(n_components=2)
    FaX_train = fa.fit_transform(FaX_train)
    FaX_test = fa.transform(FaX_test)


# In[5]:



    clf = LinearRegression()
    clf.fit(PcaX_train,Pcay_train)
    LinearPCA = {"Score":clf.score(PcaX_test,Pcay_test),"Name":"Linear Reression With PCA"}


# In[6]:


    clf = LinearRegression()
    clf.fit(FaX_train,Fay_train)
    LinearFA={"Score":clf.score(FaX_test,Fay_test),"Name":"Linear Reression With FA"}


# In[7]:


#This is inaccurate because I had to convert the dataset into ints values be cautious 
    clf = LinearRegression()
    clf.fit(LdaX_train,Lday_train)
    LinearLda= {"Score":clf.score(LdaX_test,Lday_test),"Name":"Linear Reression With LDA "}


# In[8]:


#Using sklearn instead
#Combining sigmoid function and the three dimension reduction


    NNsigmFA_reg = MLPRegressor(activation='logistic',solver='sgd',max_iter=1000)

    NNsigmFA_reg.fit(FaX_train,Fay_train)

    NNsigmFA_reg.predict(FaX_test)
    NeuralNetLogFa={"Score":NNsigmFA_reg.score(FaX_test,Fay_test)
                   ,"Name":"Neural Network Logistic Activation with Fa"}

    NNsigmPCA_reg = MLPRegressor(activation='logistic',solver='sgd',max_iter=1000)

    NNsigmPCA_reg.fit(PcaX_train,Pcay_train)

    NNsigmPCA_reg.predict(PcaX_test)
    NeuralNetLogPca=  { "Score":NNsigmPCA_reg.score(PcaX_test,Pcay_test)
                      ,"Name":"Neural Network Logistic Activation with Pca"}

    NNsigmLDA_reg = MLPRegressor(activation='logistic',solver='sgd',max_iter=1000)

    NNsigmLDA_reg.fit(LdaX_train,Lday_train)

    NNsigmLDA_reg.predict(LdaX_test)
    NeuralNetLogLda={"Score":NNsigmPCA_reg.score(LdaX_test,Lday_test)
                    ,"Name":"Neural Network Logistic Activation with Lda"}


# In[9]:


    #Combining relu function and the three dimension reduction
    NNreluFA_reg = MLPRegressor(activation='relu',solver='sgd',max_iter=1000)

    NNreluFA_reg.fit(FaX_train,Fay_train)

    NNreluFA_reg.predict(FaX_test)
    NeuralNetRelFa={"Score":NNsigmFA_reg.score(FaX_test,Fay_test)
                   ,"Name":"Neural Network Relu Activation with Fa"}


    NNreluPCA_reg = MLPRegressor(activation='relu',solver='sgd',max_iter=1000)

    NNreluPCA_reg.fit(PcaX_train,Pcay_train)

    NNreluPCA_reg.predict(PcaX_test)
    
    NeuralNetRelPca={"Score":NNsigmPCA_reg.score(PcaX_test,Pcay_test)
                    ,"Name":"Neural Network Relu Activation with Pca"}

    NNreluLDA_reg = MLPRegressor(activation='relu',solver='sgd',max_iter=1000)

    NNreluLDA_reg.fit(LdaX_train,Lday_train)

    NNreluLDA_reg.predict(LdaX_test)
    
    NeuralNetRelLda={"Score":NNsigmPCA_reg.score(LdaX_test,Lday_test)
                    ,"Name":"Neural Network Relu Activation with Lda"}


# In[10]:


#Combining tanh function and the three dimension reduction
    NNtanhFA_reg = MLPRegressor(activation='tanh',solver='sgd',max_iter=1000)

    NNtanhFA_reg.fit(FaX_train,Fay_train)

    NNtanhFA_reg.predict(FaX_test)

    NeuralNetTanFa={"Score":NNsigmFA_reg.score(FaX_test,Fay_test)
                   ,"Name":"Neural Network Tanh Activation with Fa"}
    

    NNtanhPCA_reg = MLPRegressor(activation='tanh',solver='sgd',max_iter=1000)

    NNtanhPCA_reg.fit(PcaX_train,Pcay_train)

    NNtanhPCA_reg.predict(PcaX_test)
   
    NeuralNetTanPca={"Score":NNsigmPCA_reg.score(PcaX_test,Pcay_test)
                    ,"Name":"Neural Network Tanh Activation with Pca"}

    NNtanhLDA_reg = MLPRegressor(activation='tanh',solver='sgd',max_iter=1000)

    NNtanhLDA_reg.fit(LdaX_train,Lday_train)

    NNtanhLDA_reg.predict(LdaX_test)
    
    NeuralNetTanLda={"Score":NNsigmPCA_reg.score(LdaX_test,Lday_test)
                    ,"Name":"Neural Network Tanh Activation with Lda"}
    
    Models= [LinearPCA,LinearFA,LinearLda,
            NeuralNetLogFa,NeuralNetLogPca,NeuralNetLogLda,
            NeuralNetRelFa,NeuralNetRelPca,NeuralNetRelLda,
            NeuralNetTanFa,NeuralNetTanPca,NeuralNetTanLda]
    best = Models[0]
    for mod in Models:
        print(mod)
        if(mod.get("Score")>best.get("Score")):
            best = mod
    return best
    
        

BestModel(22)

