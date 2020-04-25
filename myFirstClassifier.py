#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#data loading
data = pd.read_csv('C:\\Users\\Jhidri Achraf\\Desktop\\reduit\\TestData.csv',sep=',')


# In[3]:


data=data.iloc[0:,19:]


# In[4]:


#data.info()


# In[5]:


#data.isnull().sum()


# In[6]:


# bins = (2,5,9)
# group_names=['less','more']
# data['Classe']=pd.cut(data['Classe'],bins=bins,labels=group_names)
# data['Classe'].unique()


# In[7]:


# data.head(40).tail(20)


# In[8]:





# In[9]:


#label_Classe= LabelEncoder()
#data['Classe']=label_Classe.fit_transform(data['Classe'])#utile pour normaliser les valeurs des classes


# In[10]:


data.tail(10)


# In[11]:


data['Classe'].value_counts()


# In[30]:


sns.countplot(data['Classe'])


# In[13]:


X =data.drop('Classe',axis=1)  
Y =data['Classe']


# In[14]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=42)


# In[15]:


#see 21:43


# In[16]:


#  X_test
# Y_test


# ## Random Forest Classifier
# Définition : est une technique d'apprentissage automatique, elle se base sur une forêt d'arbre de décision
# chaque arbre de décision fournit ça propre décision , ainsi un vote est fait et la décision qu'à le plus de vote est élu comme décision final de l'arbre.

# ### Avantages
# - ils sont utilsé dans la classification et la régression
# - gére et maintient la précision pour les données manquants
# - quand le nombre d'arbres est plus grand que le nombres de donnée  dans le dataSet il est garanti  qu'il n y aura pas de sur-apprentissage a cause du system de bagging et sous-espaces aléatoires 
# - gére des grand dataset avec de grandes dimension

# ### Inconvénients
# - malgré qu'ils sont utile pour la classification , ils le sont moins que pour la régression du faite qu'il nous donne pas des valeurs pour savoir a quel pourcentage un element est de classe A ou B , en effet il répond il est de classe A ou de classe B , mais on ne sais pas a quel point.
# - on a pas de controle sur ce que fait le modèle

# ### Application
# - le secteur des banques , dans l'évaluation des risque
# - les gains/pertes dans le domaine de la bourse
# - ....

# ### mots clés 

# - Bagging : est un meta algorithm d'ensemble , qui augmente la stabilité et réduit la variance, les données sont piochées aléatoirement et mis dans des sous ensemble, a chaque entrainement le modèle utilise un sous ensemble totalement différent pour éviter le sur-apprentissage 

# ## Exemple

# In[17]:


#ici on a choisi juste 2 arbres étant donné que la taille de notre data set n'est pas grande
rfc = RandomForestClassifier(n_estimators=3) 
rfc.fit(X_train,Y_train)
pred_rfc=rfc.predict(X_test)


# In[18]:


print(pred_rfc)


# # analyse du rapport de la classification :
#      - on peut voir la précision du model, après avoir été entrainé sur notre DataSet
#      - on peut aussi voir la précision de classification de chaque classe comparé aux autres par exemple les classe de 3 a 9 sont a 100% détéctées contrairement 0 1 et 2 que le modèle les confends avec un taux pas très élever .
#      - on peut aussi voir les rappels de chaque classe c'est a dire quel proportion de résultats positif réels a été identifié correctement.
#      
#       

# In[19]:


print(classification_report(Y_test,pred_rfc))


# # analyse de la matrice de confusion

# In[20]:


print(confusion_matrix(Y_test,pred_rfc ))


# la matrice de confusion permet d'analyser la qualité d'un système de classification, comme son nom l'indique elle determine le niveau de confusion entre classe
# les lignes c'est l'entré et les colonnes c'est le résultat de la classification par exemple :
# - (0,0) => trois 0 ont été classé comme des 0 (Vrai positif)
# - (0,X) => il y a jamais eu de confusion de la classe 0 avec d'autres caractère (Vrai Négatif)
# - (X,0) => il y a eu un 8 et un 6 , réels qui ont été confondu avec 0 (Faux positifs)
# ainsi plus la matrice est proche d'une matrice diagonale plus le système de classification est bon.
#  

# # SVM classifier

# In[21]:


clf=svm.SVC()
clf.fit(X_train,Y_train)
pred_clf = clf.predict(X_test)
pred_clf


# In[22]:


print(classification_report(Y_test,pred_clf))


# In[23]:


print(confusion_matrix(Y_test,pred_clf ))


# ## Neural Networks

# In[24]:


mlpc=MLPClassifier(hidden_layer_sizes=(50,50,50,50),max_iter=300)
mlpc.fit(X_train,Y_train)
pred_mlpc=mlpc.predict(X_test)


# In[25]:


print(classification_report(Y_test,pred_mlpc))


# In[26]:


print(confusion_matrix(Y_test,pred_mlpc))


# In[27]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(Y_test,pred_mlpc)
cm


# In[28]:


Xnew = [[-0.609619,-0.63984,-0.648256,-0.727993,-0.73143,-0.729578,-0.742199,-0.416766,-0.399018,-0.33351,-0.333725,-0.308288,-0.312214,-0.306999,-0.315807,-0.298464,-0.389444,-0.445049,-0.613308,-0.658376,-0.676779,0.594535,0.597617,0.597559,0.565377,0.451438,0.429529,0.340736,0.195119,0.210317,0.181377,0.178843,0.142533,0.14941,0.106932,0.0485054,-0.0275623,-0.0688092,-0.0578532,-0.0828497,0.0240832,0.0152373]]
ynew = clf.predict(Xnew)
ynew

