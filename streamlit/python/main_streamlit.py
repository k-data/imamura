import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pydeck as pdk
import plotly.express as px 
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


conn = sqlite3.connect('database.db')
c = conn.cursor()

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_user():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_user(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def origin():
	if 'counter' not in st.session_state:
		st.session_state['counter'] = 0


def plus_one_clicks():
	st.session_state['counter'] += 1


def return_int(time):
    if time == '9:':
        return int(9)
    elif time == '7:':
        return int(7)
    elif time == '6:':
        return int(6)
    elif time == '8:':
        return int(8)
    elif time == '4:':
        return int(4)
    elif time == '0.':
        return int(0)
    elif time == '3:':
    	return int(3)
    else:
        return int(time)




menu = ["ログイン","サインアップ"]

choice = st.sidebar.selectbox("メニュー",menu)


if choice == "サインアップ":
	st.subheader("新しいアカウントを作成します")
	new_user = st.text_input("ユーザー名を入力してください")
	new_password = st.text_input("パスワードを入力してください",type='password')

	if st.button("サインアップ"):
		create_user()
		add_user(new_user,make_hashes(new_password))
		st.success("アカウントの作成に成功しました")
		st.info("ログイン画面からログインしてください")

elif choice == "ログイン":
	st.info("please_login")

	username = st.sidebar.text_input("ユーザー名を入力してください")
	password = st.sidebar.text_input("パスワードを入力してください",type='password')
	if st.sidebar.checkbox("ログイン"):
		create_user()
		hashed_pswd = make_hashes(password)

		result = login_user(username,check_hashes(password,hashed_pswd))
		if result:

			st.success("{}さんでログインしました".format(username))

		else:
			st.warning("ユーザー名かパスワードが間違っています")

		if result :
			custmer = ['三機工業', 'メタウォーター']
			select = ['売上', '仕入れ', '数量']
			choice = st.sidebar.selectbox('選択してください', custmer)	

			if choice == custmer[0]:
				menu = st.sidebar.selectbox('項目を選択してください', select)

				if menu == select[0]:
					df = pd.read_csv('streamlit/data/sankikougyou.csv')
					df = df.dropna()
					df['year'] = df['日付'].apply(lambda x: int(x[:4]))
					df['month'] = df['日付'].apply(lambda x : int(x[5:7]))
					df['day'] = df['日付'].apply(lambda x: int(x[-2:]))
					year_list = [i for i in df['year'].unique()]
					years = st.slider('年を選択してください', min_value=2020, max_value=2022, step=1, value=2020)

					if years == year_list[0]:
						df_year = df[df['year'] == 2020]
						df_month = df_year.groupby('month').sum()
						df_revenue = df_month[['金額']]
						st.bar_chart(df_revenue)
						st.write('↓月ごとの売上金額')
						st.write(df_revenue.T)

					elif  years == year_list[1]:
						df_year = df[df['year'] == 2021]
						df_month = df_year.groupby('month').sum()
						df_revenue = df_month[['金額']]
						st.bar_chart(df_revenue)
						st.write('↓月ごとの売上金額')
						st.write(df_revenue.T)

					else:
						df_year = df[df['year'] == 2022]
						df_month = df_year.groupby('month').sum()
						df_revenue = df_month[['金額']]
						st.bar_chart(df_revenue)
						st.write('↓月ごとの売上金額')
						st.write(df_revenue.T)
					


					con = st.checkbox('all')
					if con == True:

						df_one = df[df['year'] == 2020]
						df_month = df_one.groupby('month').sum()
						df_revenue_1 = df_month[['金額']]
						df_revenue_1 = df_revenue_1.rename(columns={'金額': 2020})
						df_revenue_1['月'] = df_revenue_1.index


						df_two = df[df['year'] == 2021]
						df_month = df_two.groupby('month').sum()
						df_revenue_2 = df_month[['金額']]
						df_revenue_2 = df_revenue_2.rename(columns={'金額':2021})
						df_revenue_2['月'] = df_revenue_2.index

						df_three = df[df['year'] == 2022]
						df_month = df_three.groupby('month').sum()
						df_revenue_3 = df_month[['金額']]
						df_revenue_3 = df_revenue_3.rename(columns={'金額':2022})
						df_revenue_3['月'] = df_revenue_3.index

						df_all = df_revenue_1.merge(df_revenue_2, how='outer', on='月')
						df_all = df_all.merge(df_revenue_3, how='outer', on='月')
						df_all.index = df_all['月'].values
						df_all2 = df_all[[2020, 2021, 2022]]
						st.line_chart(df_all2)
						df_year_sum = df.groupby('year').sum()
						with st.container():
							col1, col2 = st.columns([1, 1])
						
						with col1:
							st.write(f'年別の{menu}比較')
							fig, ax = plt.subplots(figsize=(6, 4))
							x = df_year_sum.index
							y = df_year_sum['金額'].values
							sns.barplot(x, y)
							st.pyplot(fig)
							st.write('各年の合計金額')
							st.write(df_year_sum.loc[:,'金額'])

						with col2:
							st.write(f'月別の{menu}比較')

							x = df_revenue_1['月'].values
							height = df_revenue_1[2020].values
							
							x2 = df_revenue_2['月'].values
							height2 = df_revenue_2[2021].values

							x3 = df_revenue_3['月'].values
							height3 = df_revenue_3[2022].values

							fig, axis = plt.subplots(figsize=(8, 6))
							axis.bar(x, height, label='2020', align='edge')
							axis.bar(x2, height2, label='2021', align='center')
							axis.bar(x3, height3, label='2022', align='edge')
							plt.xticks(np.arange(1, 13, 1))
							plt.legend()
							st.pyplot(fig)

						column_list = ['m3', 'kg', '台']
						st.write(f'単位別の{menu}を調べる')
						column = st.selectbox('選択してください', column_list)
						
						for i in range(len(column_list)):
							if column == column_list[i]:
								df_s = df[df['単位'] == column]
						
						max_value = 2022
						min_value = 2020
						product_y = st.slider('年を選択してくだい', min_value=min_value, max_value=max_value, step=1, value=min_value)
						
						for i in range(len(year_list)):
							if product_y == year_list[i]:
								df_s = df_s[df_s['year'] == product_y]

						df_s_m = df_s.groupby('month').sum()
						df_sm = df_s_m.loc[:, '金額']
						st.write(f'{product_y}年の {column}:売上')
						st.bar_chart(df_sm)

						a_all = st.checkbox(f'年別{column}の売上比較')
						if a_all == True:
							df_a_1 = df[df['year'] == year_list[0]]
							df_a_2 = df[df['year'] == year_list[1]]
							df_a_3 = df[df['year'] == year_list[2]]
							df_a_1 = df_a_1.groupby('単位').sum()
							df_a_2 = df_a_2.groupby('単位').sum()
							df_a_3 = df_a_3.groupby('単位').sum()

							df_a_1['単位'] = df_a_1.index
							df_a_2['単位'] = df_a_2.index
							df_a_3['単位'] = df_a_3.index

							df_a_1 = df_a_1.loc[f'{column}', ['金額']].values
							df_a_2 = df_a_2.loc[f'{column}', ['金額']].values
							df_a_3 = df_a_3.loc[f'{column}', ['金額']].values

							df_a_all = pd.DataFrame({2020: df_a_1, 2021: df_a_2, 2022: df_a_3})
							df_a_all = df_a_all.T
							st.bar_chart(df_a_all)
							st.write(f'{column}の売上金額↓')
							st.write(df_a_all)



				elif menu == select[1]:
					st.write('not data')
					
				else:
					df = pd.read_csv('streamlit/data/sankikougyou.csv')
					df = df.dropna()
					df['year'] = df['日付'].apply(lambda x: int(x[:4]))
					df['month'] = df['日付'].apply(lambda x : int(x[5:7]))
					df['day'] = df['日付'].apply(lambda x: int(x[-2:]))
					year_list = [i for i in df['year'].unique()]
					column_list = ['m3', 'kg', '台']
					st.write(f'単位別の{menu}を調べる')
					column = st.selectbox('選択してください', column_list)
					for i in range(len(column_list)):
						if column == column_list[i]:
							df_val = df[df['単位'] == column]	
					
					years = st.slider('年を選択してください', min_value=2020, max_value=2022, step=1, value=2020)

					if years == year_list[0]:
						df_year = df_val[df_val['year'] == 2020]
						df_month = df_year.groupby('month').sum()
						df_value = df_month[['数量']]
						st.bar_chart(df_value)
						st.write('↓月ごとの数量')
						st.write(df_value.T)

					elif  years == year_list[1]:
						df_year = df_val[df_val['year'] == 2021]
						df_month = df_year.groupby('month').sum()
						df_value = df_month[['数量']]
						st.bar_chart(df_value)
						st.write('↓月ごとの数量')
						st.write(df_value.T)

					else:
						df_year = df_val[df_val['year'] == 2022]
						df_month = df_year.groupby('month').sum()
						df_value = df_month[['数量']]
						st.bar_chart(df_value)
						st.write('↓月ごとの数量')
						st.write(df_value.T)
					


					con = st.checkbox('all')
					if con == True:

						df_one = df[df['year'] == 2020]
						df_one = df_one[df_one['単位'] == column]
						df_month = df_one.groupby('month').sum()
						df_value_1 = df_month[['数量']]
						df_value_1 = df_value_1.rename(columns={'数量': 2020})
						df_value_1['月'] = df_value_1.index


						df_two = df[df['year'] == 2021]
						df_two = df_two[df_two['単位'] == column]
						df_month = df_two.groupby('month').sum()
						df_value_2 = df_month[['数量']]
						df_value_2 = df_value_2.rename(columns={'数量':2021})
						df_value_2['月'] = df_value_2.index

						df_three = df[df['year'] == 2022]
						df_three = df_three[df_three['単位'] == column]
						df_month = df_three.groupby('month').sum()
						df_value_3 = df_month[['数量']]
						df_value_3 = df_value_3.rename(columns={'数量':2022})
						df_value_3['月'] = df_value_3.index

						df_all = df_value_1.merge(df_value_2, how='outer', on='月')
						df_all = df_all.merge(df_value_3, how='outer', on='月')
						df_all.index = df_all['月'].values
						df_all2 = df_all[[2020, 2021, 2022]]
						st.line_chart(df_all2)
						df_val = df[df['単位'] == column]
						df_year_sum = df_val.groupby('year').sum()
						with st.container():
							col1, col2 = st.columns([1, 1])
						
						with col1:
							st.write(f'年別の{column}数量比較')
							fig, ax = plt.subplots(figsize=(6, 4))
							x = df_year_sum.index
							y = df_year_sum['数量'].values
							sns.barplot(x, y)
							st.pyplot(fig)
							st.write('各年の合計')
							st.write(df_year_sum.loc[:,'数量'])

						with col2:
							st.write(f'月別の{column}比較')

							x = df_value_1['月'].values
							height = df_value_1[2020].values
							
							x2 = df_value_2['月'].values
							height2 = df_value_2[2021].values

							x3 = df_value_3['月'].values
							height3 = df_value_3[2022].values

							fig, axis = plt.subplots(figsize=(8, 6))
							axis.bar(x, height, label='2020', align='edge')
							axis.bar(x2, height2, label='2021', align='center')
							axis.bar(x3, height3, label='2022', align='edge')
							plt.xticks(np.arange(1, 13, 1))
							plt.legend()
							st.pyplot(fig)

