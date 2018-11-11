# In[]
import requests
from bs4 import BeautifulSoup as bs
import re
data=[]
# In[]
def getcoded_df(df,make_context,model_context):
    for i in range(len(df["make"])):
        for j in range(len(make_context)):
            if df["make"][i]==make_context[j]:
                df["make"][i]=j

    for i in range(len(df["model"])):
        for j in range(len(model_context)):
            if df["model"][i]==model_context[j]:
                df["model"][i]=j
    return df

def getcontext(df):
    make_context=[]
    model_context=[]
    for i in range(len(df["make"])):
        if df["make"][i] not in make_context:
            make_context.append(df["make"][i])
    for i in range(len(df["model"])):
        if df["model"][i] not in model_context:
            model_context.append(df["model"][i])
    return make_context,model_context
# In[]
import mysql.connector
cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='bama_data_ml')
cursor=cnx.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS train_cars(make nvarchar(50), model nvarchar(50), kms int, price bigint, year int)')
cursor.execute('CREATE TABLE IF NOT EXISTS test_cars(make nvarchar(50), model nvarchar(50), kms int, price bigint, year int)')
cursor.execute('delete from train_cars')
cursor.execute('delete from test_cars')
cnx.commit()
for i in range(1,100):
    r=requests.get('https://bama.ir/car?/page='+str(i))
    
    soup=bs(r.text,'html.parser')
    kms=soup.find_all('p',attrs={'class':'price hidden-xs'})
    price=soup.find_all('p',attrs={'class':'cost'})
    temp=soup.find_all('h2',attrs={'class':'persianOrder'})
    
    year=[]
    make=[]
    model=[]
    for n in temp:
        x=re.sub(r'\s{2,}','',n.text)
    for n in temp:
        year.append(re.findall(r'(^.+?)\،',x))
        make.append(re.findall(r'\،(.+?)\،',x))
        model.append(re.findall(r'^[^،]*،[^،]*،(.*)',x))
    
    edited_price=[]
    edited_km=[]
    edited_name=[]
    
    for i in range(len(price)):
        if re.search(r'\d{3,}',price[i].text)== None:
            edited_price.append('0')
        else:
            temp1=re.sub(r',','',price[i].text)
            edited_price.append(re.findall(r'\s(\d.*?)\s',temp1))
        if re.search(r'\d.+',kms[i].text)== None:
            edited_km.append('0')
        else:
            temp2=re.sub(r',','',kms[i].text)
            edited_km.append(re.findall(r'\s(\d.*?)\s',temp2))
        if int(year[i][0])<1500:
            temp3=int(year[i][0])+621
        else: 
            temp3=int(year[i][0])
        print(int(edited_price[i][0]))
        cursor.execute("insert into train_cars values(%s,%s,%s,%s,%s)", (make[i][0], model[i][0], int(edited_km[i][0]), int(edited_price[i][0]),temp3 ))
        cnx.commit()
cnx.close()
   # In[]:
import mysql.connector
cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='bama_data_ml')
cursor=cnx.cursor()
import pandas as pd
df_all= pd.read_sql('select * from train_cars;', con=cnx) 
make_context,model_context=getcontext(df_all)
cursor.execute("INSERT INTO test_cars SELECT * FROM train_cars WHERE price=0")
cursor.execute("delete from train_cars WHERE price=0")
cnx.commit()
df_train = pd.read_sql('select * from train_cars;', con=cnx) 
df_train_b=pd.DataFrame.copy(df_train)
df_train=getcoded_df(df_train,make_context,model_context)
df_test = pd.read_sql('select * from test_cars;', con=cnx) 
df_test_b=pd.DataFrame.copy(df_test)
df_test=getcoded_df(df_test,make_context,model_context)
cnx.close()
# In[]
'''
import numpy as np
from sklearn import preprocessing
x=np.transpose(df_cars.values)
x_scaled = preprocessing.normalize(x, norm='l1')
df = pd.DataFrame(np.transpose(x_scaled))
'''
# In[]
df_train["kms"]=df_train["kms"]*(10**-3)
df_test["kms"]=df_test["kms"]*(10**-3)
# In[]
df3=pd.get_dummies(df_train["model"],prefix=['model'])
df_train_1hot=pd.concat([df3, df_train],axis=1)
del df_train_1hot["model"]
df4=pd.get_dummies(df_train_1hot["make"],prefix=['make'])
df_train_1hot=pd.concat([df4, df_train_1hot],axis=1)
del df_train_1hot["make"]

df3=pd.get_dummies(df_test["model"],prefix=['model'])
df_test_1hot=pd.concat([df3, df_test],axis=1)
del df_test_1hot["model"]
df4=pd.get_dummies(df_test_1hot["make"],prefix=['make'])
df_test_1hot=pd.concat([df4, df_test_1hot],axis=1)
del df_test_1hot["make"]
# In[]
from sklearn.kernel_ridge import KernelRidge
n_samples, n_features = len(df_train_1hot), len(df_train_1hot.columns)-1
df_input=df_train_1hot.loc[:, df_train_1hot.columns != 'price']
y = df_train_1hot["price"]
X = df_input
clf = KernelRidge(alpha=1.0)
clf.fit(X, y) 
df_input_test=df_test_1hot.loc[:, df_test_1hot.columns != 'price']
clf.predict(df_input_test.loc[[1]])