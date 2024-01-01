import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score

st.title('Aplikasi Clustering Nasabah Kartu Kredit')

st.write('''
    Aplikasi ini dibuat untuk menyediakan informasi yang dapat membantu perusahaan kartu kredit dalam memahami nasabah mereka 
    dan merancang strategi yang efektif untuk mempertahankan para nasabahnya. So let's dive in!
    ''')

df = pd.read_csv("BankChurners.csv")
st.header("Data Asli")
show_rawdata = st.checkbox('Tampilkan Data Asli? ')
if show_rawdata: 
    st.write(df)

df = df.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)

numerical = []
catgcols = []

for col in df.columns:
    if df[col].dtype in ['int64' , 'float64']:
      numerical.append(col)
    else:
      catgcols.append(col)

for col in df.columns:
    if col in numerical:
      df[col].fillna(df[col].median(), inplace=True)
    else:
      df[col].fillna(df[col].mode()[0], inplace=True)

LE=LabelEncoder()
for i in catgcols:
    df[i]=df[[i]].apply(LE.fit_transform)

sub_df = df[['Customer_Age', 'Dependent_count', 'Education_Level',
            'Marital_Status', 'Income_Category', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
            'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']]

scaler = StandardScaler()
scaler.fit(sub_df)
scaled_sub_df = pd.DataFrame(scaler.transform(sub_df),columns= sub_df.columns )

wcss=[]
for i in range (1,11):
  model = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=1000, random_state=42)
  model.fit(sub_df)
  wcss.append(model.inertia_)


st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)

st.sidebar.subheader("Elbow: ")
show_elbow = st.sidebar.checkbox("Hitung elbow? ")
if show_elbow:
   Elbow_M = KElbowVisualizer(KMeans(), k=10)
   Elbow_M.fit(sub_df)
   Elbow_M.show()
   st.set_option('deprecation.showPyplotGlobalUse', False)
   elbo_plot = st.sidebar.pyplot()

def k_means(n_clust):
   kmeans = KMeans(n_clusters=n_clust).fit(sub_df)
   sub_df['Cluster'] = kmeans.labels_
   st.header('Cluster Plot')


   plt.figure(figsize=(10, 6))
   sns.scatterplot(data=sub_df, x='Total_Trans_Amt', y='Total_Trans_Ct', hue='Cluster', palette='viridis', s=50)
   plt.title('Scatter Plot of Total_Trans_Amt vs. Total_Trans_Ct')
   plt.xlabel('Total Transaction Amount')
   plt.ylabel('Total Transaction Count')
   st.pyplot()

   plt.figure(figsize=(10, 6))
   sns.scatterplot(data=sub_df, x='Months_on_book', y='Total_Trans_Amt', hue='Cluster', palette='viridis', s=50)
   plt.title('Scatter Plot of Months_on_book vs. Total_Trans_Amt')
   plt.xlabel('Months_on_book')
   plt.ylabel('Total_Trans_Amt')

   st.pyplot()
   
   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='Total_Trans_Amt', y='Contacts_Count_12_mon', data=sub_df, hue='Cluster', palette='viridis', s=50)
   plt.title('Scatter Plot of Total_Trans_Amt vs. Contacts_Count_12_mont')
   plt.xlabel('Total Transaction Amount')
   plt.ylabel('Contacts Count (12 Months)')
   plt.legend(title='Cluster')
   st.pyplot()

   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='Customer_Age', y='Credit_Limit', data=sub_df, hue='Cluster', palette='viridis', s=50)
   plt.title('Scatter Plot of Customer_Age vs. Credit_Limit')
   plt.xlabel('Customer Age')
   plt.ylabel('Credit Limit')
   plt.legend(title='Cluster')
   st.pyplot()
   
   st.header('Data Setelah Clustering')
   st.write(sub_df)

k_means(clust)
