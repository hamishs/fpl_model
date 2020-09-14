import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#local
import optimiser as opt

predictions = pd.read_csv('results.csv')

predictions = predictions.rename(columns = {
	'season' : 'season_name',
	'team_code' : 'team_id'})

predictions['position'] = predictions['element_type'].map({
	1 : 'goalkeeper',
	2 : 'defender',
	3 : 'midfielder',
	4 : 'forward'
	})

df_list = []
for name in predictions.name.unique():
	df = predictions.loc[predictions.name == name]
	df['total_points'] = df['total_points'].shift(-1)
	df_list.append(df)

predictions = pd.concat(df_list)
predictions = predictions.fillna(0)


all_gws = {}
gw_predictions = pd.DataFrame()
for GW in range(38):
	#select optimum squad for each gw

	print('Selecting team for gw {}'.format(GW+1))

	single_gw = predictions.loc[predictions.GW == GW]

	(optimal_squad, solution_info) = opt.get_optimal_squad(
		single_gw, formation = None, budget = 83.5, season = '2019/20',
		optimise_on='exp_points', monitor = 'total_points')

	capt = max(optimal_squad, key = lambda x:x['exp_points'])

	all_gws[GW] = (optimal_squad, solution_info)

	solution_info['points_next_w_c'] = solution_info['total_points'] + capt['total_points']
	solution_info['GW'] = GW

	print(solution_info['points_next_w_c'])

	gw_predictions = gw_predictions.append(solution_info, ignore_index = True)


print(gw_predictions)

total_points = 0
for GW in range(38):
	total_points += all_gws[GW][1]['points_next_w_c']

print(total_points)

gw_predictions.plot(x = 'GW', y = 'actual')
plt.show()









