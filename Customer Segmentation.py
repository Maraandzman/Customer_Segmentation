#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sqlalchemy import types
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import cx_Oracle


# In[6]:


dsn_tns = cx_Oracle.makedsn(DSN_name, Port, service_name=service_name) # if needed, place an 'r' before any parameter in order to address special characters such as '\\'.\n"
conn = cx_Oracle.connect(user=user, password=password, dsn=dsn_tns) # if needed, place an 'r' before any parameter in order to address special characters such as '\\'. For example, if your user name contains '\\', you'll need to place 'r' before the user name: user=r'User Name


# In[7]:


query = """ select * from lau_s.segment_analysis where last_recharge_date between '01/FEB/20' and '29/FEB/20' """
df_ora = pd.read_sql(query, con=conn)
df_ora.head()


# In[8]:


df_ora.head()


# In[11]:


import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.offline as pyoff


# In[12]:


pyoff.init_notebook_mode()


# In[13]:


df_ora['LAST_RECHARGE_DATE'] = pd.to_datetime(df_ora['LAST_RECHARGE_DATE'])


# In[14]:


df_ora.info()


# In[15]:


#Separate the months

df_202002 = df_ora[(df_ora['LAST_RECHARGE_DATE'] > '2020-01-31') & (df_ora['LAST_RECHARGE_DATE'] < '2020-03-01')]
df_202002.head()


# In[16]:


df_202002.info()


# In[17]:


df_user = pd.DataFrame(df_202002['MSISDN'].unique())
df_user.columns = ['MSISDN']

#get the max purchase date for each customer and create a dataframe with it
df_max_purchase = df_202002.groupby('MSISDN').LAST_RECHARGE_DATE.max().reset_index()
df_max_purchase.columns = ['MSISDN', 'MaxPurchaseDate']


# In[18]:


#we take our observation point as the max invoice date in our dataset
df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
df_user = pd.merge(df_user, df_max_purchase[['MSISDN', 'Recency']], on ='MSISDN')


# In[19]:


#plot a recency histogram
plot_data = [
        go.Histogram(
        x=df_user['Recency']
        )
    ]
plot_layout = go.Layout(
                title='Recency')
              
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
    


# In[20]:


df_user.Recency.describe()


# In[21]:


from sklearn.cluster import KMeans
sse={}
df_recency = df_user[['Recency']]
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()),list(sse.values()))
plt.xlabel("Number of clusters")
plt.show()


# In[22]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[["Recency"]])
df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])


# In[23]:


#function for ordering clsuter numbers
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


                                        
                                        


# In[24]:


df_user = order_cluster('RecencyCluster', 'Recency', df_user,True)


# In[25]:


df_user.groupby('RecencyCluster').describe()


# In[26]:


#get order counts for each user and create a dataframe with it
tx_frequency = df_202002.groupby('MSISDN').RCH_COUNT_VOUCHER.sum().reset_index()
tx_frequency.columns = ['MSISDN','Frequency']

#add this data to our main dataframe
df_user = pd.merge(df_user, tx_frequency, on='MSISDN')


# In[27]:


#plot the histogram
plot_data = [
    go.Histogram(
        x=df_user.query('Frequency < 30')['Frequency'], nbinsx=10
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[28]:


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Frequency']])
df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])

#order the frequency cluster
df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,True)

#see details of each cluster
df_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[29]:


#calculate revenue for each customer
tx_revenue = df_202002.groupby('MSISDN').ASPU.sum().reset_index()
tx_revenue.columns = ['MSISDN','Revenue']

#merge it with our main dataframe
df_user = pd.merge(df_user, tx_revenue, on='MSISDN')


# In[35]:


#plot the histogram
plot_data = [
    go.Histogram(
        x=df_user.query('Revenue < 400')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[31]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_user[['Revenue']])
df_user['RevenueCluster'] = kmeans.predict(df_user[['Revenue']])

#order the cluster numbers
df_user = order_cluster('RevenueCluster', 'Revenue',df_user,True)

#show details of the dataframe
df_user.groupby('RevenueCluster')['Revenue'].describe()


# In[32]:


#calculate overall score and use mean() to see details
df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['RevenueCluster']
df_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[33]:


df_user.groupby('OverallScore')['Recency','Frequency','Revenue'].count()


# In[36]:


df_user.head()


# In[37]:


low_value = df_user.loc[df_user['OverallScore'] <= 1]
mid_value = df_user.loc[df_user['OverallScore'].isin([2,3])]
high_value = df_user.loc[df_user['OverallScore'] > 3]


# In[70]:


low_value.to_excel('C:\\Users\\Bukasa_r\\Desktop\\low_value.xlsx')


# In[71]:


mid_value.to_excel('C:\\Users\\Bukasa_r\\Desktop\\mid_value.xlsx')


# In[72]:


high_value.to_excel('C:\\Users\\Bukasa_r\\Desktop\\high_value.xlsx')


# In[39]:


#create sample to process scatter graph
df2 = pd.read_excel('C:\\Users\\bukasa_r\\Desktop\\Reports\\Ad-Hoc\\tester1.xlsx')


# In[41]:



df2['Segment'] = 'Low-Value'
df2.loc[df2['OverallScore']>1,'Segment'] = 'Mid-Value' 
df2.loc[df2['OverallScore']>3,'Segment'] = 'High-Value' 


# In[42]:


df2.groupby('Segment').count()


# In[43]:


# Revenue vs Frequency
tx_graph = df2.query("Revenue < 1000 and Frequency < 30")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'red',
            opacity= 0.7
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'orange',
            opacity= 1
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.7
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#Revenue Recency

tx_graph = df2.query("Revenue < 1000 and Recency < 30")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'red',
            opacity= 1
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'orange',
            opacity= 0.7
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'green',
            opacity= 0.7
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

# Revenue vs Frequency
tx_graph = df2.query("Frequency < 30 and Recency < 30")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'red',
            opacity= 1
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        mode='markers',
        name='Mid',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'orange',
            opacity= 0.7
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        mode='markers',
        name='High',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

