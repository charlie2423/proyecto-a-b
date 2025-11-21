import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import scipy.stats as stats
import datetime as dt
from statsmodels.stats.proportion import proportions_ztest
df_marketing_events = pd.read_csv('/datasets/ab_project_marketing_events_us.csv')
df_final_users = pd.read_csv('/datasets/final_ab_new_users_upd_us.csv',parse_dates=['first_date'])
df_final_events = pd.read_csv('/datasets/final_ab_events_upd_us.csv',parse_dates=['event_dt'])
df_final_participants = pd.read_csv('/datasets/final_ab_participants_upd_us.csv')
users = df_final_participants.merge(df_final_users, on= 'user_id',how='left')
df_union = df_final_events.merge(users[['user_id','first_date','group','region']],on='user_id',how='inner')
df_union = df_union[df_union['event_dt']<=df_union['first_date'] + pd.Timedelta(days=14)]
df_union = df_union[df_union['region']=='EU']

print(df_union)

user_events = (
    df_union
    .groupby(['user_id', 'group', 'event_name'])
    .agg({'event_dt': 'count'})  
    .reset_index()
    .pivot_table(index=['user_id', 'group'],
                 columns='event_name',
                 values='event_dt',
                 fill_value=0)
    .reset_index()
)

user_events['page'] = (user_events['product_page']>0).astype(int)
user_events['cart']=(user_events['product_cart']>0).astype(int)
user_events['purchase_']=(user_events['purchase']>0).astype(int)

group_sum = user_events.groupby('group').agg(
    user_total = ('user_id','nunique'),
    num_page=('page','sum'),
    num_cart=('cart','sum'),
    num_purchase = ('purchase_','sum'),
).reset_index()
group_sum['rate_page'] = group_sum['num_page']/group_sum['user_total']
group_sum['rate_page_rate_cart'] = group_sum['num_cart'] / group_sum['num_page']
group_sum['rate_cart_rate_pucharse'] = group_sum['num_purchase']/group_sum['num_cart']
print(group_sum)

num_events = df_union.groupby(['group','user_id']).agg({'event_name':'count'}).reset_index()
num_events.rename(columns={'event_name':'num_events'},inplace=True)
print(num_events)

plt.figure(figsize=(15,15))
sns.histplot(data=num_events,x='num_events',hue='group',bins=30,kde=True)
plt.title('distribucion de eventos por usarios entre grupo')
plt.xlabel('numero de eventos')
plt.ylabel('frecuencia')
plt.show()

A_value = num_events[num_events['group']=='A']['num_events']
B_value = num_events[num_events['group']=='B']['num_events']

stat,p = mannwhitneyu(A_value,B_value,alternative='two-sided')
print(f"mannwhitneyu test = {stat},p-valor={p}")
alpha=0.5
if p < alpha:
    print('no hay diferencia significativa')
else:
    print('hay diferencia significativa')

    usarios_aa =  set(df_union[df_union['group']=='A']['user_id'].unique())
usarios_bb =  set(df_union[df_union['group']=='B']['user_id'].unique())

usarios_en_AB = usarios_aa.intersection(usarios_bb)

if usarios_en_AB:
    print('numero de usarios en ambas prueabs', len(usarios_en_AB))
    print(list(usarios_en_AB))
else:
    print('no hay usarios presentes en las muestras')

df_union['day'] = df_union['event_dt'].dt.date

grupo_dia = df_union.groupby('day')['event_name'].count()
print(grupo_dia)


plt.figure(figsize=(15,15))
grupo_dia.plot(kind='bar')
plt.title('distribucion de los eventos por dia')
plt.xlabel('dia del mes')
plt.ylabel('numero de eventos')
plt.grid(True)
plt.show()

df_union['event_dt']  = pd.to_datetime(df_union['event_dt'])

df_filtrado_fecha =df_union[(df_union['event_dt']>='2020-12-07')&(df_union['event_dt']<='2020-12-21')]

df_clean = df_filtrado_fecha[~df_filtrado_fecha['user_id'].isin(usarios_en_AB)]

purchases = df_clean[df_clean['event_name']=='purchase']

purchases_group = purchases.groupby('group')['user_id'].nunique()

users_total_group = df_clean.groupby('group')['user_id'].nunique()

purchases_z = purchases_group.loc[['A','B']].values

users_z = users_total_group.loc[['A','B']].values


z_stat,pval = proportions_ztest(count=purchases_z,nobs=users_z)


print('estadistico z:',z_stat)
print('valor p:',pval)

if pval < 0.05:
    print('la diferencia entre las proporciones es significativa')
else:
    print("No hay evidencia suficiente para afirmar que las proporciones son diferentes.")

product_page = df_clean[df_clean['event_name']=='product_page']
product_page_group = product_page.groupby('group')['user_id'].nunique()
product_page_z = product_page_group.loc[['A','B']].values

z_stat,pval = proportions_ztest(count=product_page_z,nobs=users_z)


print('estadistico z:',z_stat)
print('valor p:',pval)

if pval < 0.05:
    print('la diferencia entre las proporciones es significativa')
else:
    print("No hay evidencia suficiente para afirmar que las proporciones son diferentes.")

product_cart = df_clean[df_clean['event_name']=='product_cart']
product_cart_group = product_cart.groupby('group')['user_id'].nunique()
product_cart_z = product_cart_group.loc[['A','B']].values
z_stat,pval = proportions_ztest(count=product_cart_z,nobs=users_z)


print('estadistico z:',z_stat)
print('valor p:',pval)

if pval < 0.05:
    print('la diferencia entre las proporciones es significativa')
else:
    print("No hay evidencia suficiente para afirmar que las proporciones son diferentes.")