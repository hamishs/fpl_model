import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#local
import mixture

class BuildDataset():

	def __init__(self, filename, encoding = 'utf-8'):
		#add data
		self.df = pd.read_csv(filename, encoding = encoding)
		self.add_gw_0(self.df)
		self.shift_columns(self.df)
		self.rolling_cols(self.df)

	def add_gw_0(self, df):
		#rename post lockdown gws
		df['GW'] = df['GW'].apply(lambda x: x - 9 if x>38 else x)

		#add gw0 rows
		df['is_gw_0'] = False
		gw0 = df.loc[df.GW == 1]
		gw0.loc[:,['assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
			'goals_conceded', 'goals_scored', 'ict_index', 'influence',
			'minutes', 'own_goals', 'penalties_missed', 'penalties_saved',
			'red_cards', 'saves', 'threat', 'total_points', 'yellow_cards']] = 0
		gw0.loc[:,['fixture', 'kickoff_time', 'opponent_team', 'team_a_score',
			'team_h_score', 'was_home']] = np.nan
		gw0.loc[:,['round', 'GW']] = 0
		gw0.loc['is_gw_0'] = True
		df = pd.concat([gw0, df])
		self.df = df

	def shift_columns(self, df):
		#sort by player then by gw/match
		df = df.sort_values(['name', 'GW', 'kickoff_time'], ascending = True)

		#shift columns to provide observable next gw info
		shift_cols = ['opponent_team', 'selected', 'transfers_balance',
			'transfers_in', 'transfers_out', 'was_home']
		shift_cols_name = ['next_' + col for col in shift_cols]
		df_list = []
		for name in df.name.unique():
			p = df.loc[df.name == name]
			p[shift_cols_name] = p[shift_cols].shift(-1)
			df_list.append(p)
		df = pd.concat(df_list)
		self.df = df

	def rolling_cols(self, df):
		#sort by player then by gw/match
		df = df.sort_values(['name', 'GW', 'kickoff_time'], ascending = True)

		#shift columns to provide observable next gw info
		rolling_cols = ['minutes', 'ict_index', 'influence', 'bps']
		short_cols_name = ['ma_s_' + col for col in rolling_cols]
		long_cols_name = ['ma_l_' + col for col in rolling_cols]
		df_list = []
		for name in df.name.unique():
			p = df.loc[df.name == name]
			p[short_cols_name] = p[rolling_cols].rolling(5, min_periods = 1).mean()
			p[long_cols_name] = p[rolling_cols].rolling(40, min_periods = 1).mean()
			df_list.append(p)
		df = pd.concat(df_list)
		self.df = df





