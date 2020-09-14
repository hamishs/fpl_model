import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#local
import mixture

#load data
filename = 'full_dataset.csv'
df = pd.read_csv(filename)

#sort by player then by gw/match
df = df.sort_values(['season', 'name', 'GW', 'kickoff_time'], ascending = True)

#remove players with < 5 points in the season
name_seasons = df.groupby(['season', 'name']).sum()['total_points'].reset_index()
name_seasons = name_seasons.loc[name_seasons.total_points >= 5]
df = df.merge(name_seasons[['name','season']], how = 'left',
	on = ['name','season'], indicator = 'Exist')
df = df.loc[df.Exist == 'both', df.columns != 'Exist']

#fit mixture model to cluster points
K = 4
points = df['total_points'].to_numpy()
mixture_model = mixture.PoissonMixtureModel(y = points, K = K, msg = False)
clusters = mixture_model.cluster(points)
df['cluster'] = clusters

#select lambdas used in clusters
lambdas = mixture_model.lambda_m[np.in1d(list(range(K)), df.cluster.sort_values().unique())]

#shift columns to provide observable next gw info
df_list = []
for i, row in name_seasons.iterrows():
	p = df.loc[(df.name == row['name'])&(df.season == row['season'])]
	p.loc[:,'next_cluster'] = p['cluster'].shift(-1)
	df_list.append(p)
df = pd.concat(df_list)

#select columns to use for modelling
model_cols = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
	'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
	'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
	'red_cards', 'saves', 'selected', 'threat', 'total_points',
	'transfers_balance', 'transfers_in', 'transfers_out', 'value',
	'yellow_cards', 'GW', 'is_gw_0', 'cluster', 'next_selected',
	'next_transfers_balance', 'next_transfers_in',
	'next_transfers_out', 'next_was_home', 'ma_s_minutes',
	'ma_s_ict_index', 'ma_s_influence', 'ma_s_bps', 'ma_l_minutes',
	'ma_l_ict_index', 'ma_l_influence', 'ma_l_bps', 'xG', 'xA', 'xGChain',
	'xGBuildup', 'prev_1_total_points', 'prev_2_total_points',
	'prev_3_total_points', 'prev_4_total_points', 'prev_5_total_points',
	'ma_s_xG', 'ma_s_xA', 'ma_s_xGChain', 'ma_s_xGBuildup', 'ma_l_xG',
	'ma_l_xA', 'ma_l_xGChain', 'ma_l_xGBuildup', 'element_type']

train = df.loc[df.season != '2019/20', model_cols + ['next_cluster']]
train = train.dropna()

#column to predict is next cluster
X = train.drop('next_cluster', axis = 1)
y = train['next_cluster']

#over sample for balanced proportions of clusters
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

X_train = X
y_train = y

test = df.loc[df.season == '2019/20', model_cols + ['next_cluster', 'name', 'season', 'team_code']]
test = test.dropna()
test_result_data = test[['name', 'season', 'team_code', 'GW',
	'element_type', 'value', 'total_points']]
test = test.drop(['name', 'season', 'team_code'], axis = 1)
X_test = test.drop('next_cluster', axis = 1)
y_test = test['next_cluster']

#preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
	n_estimators = 22,
	max_depth = 8,
	max_features = 9)
model.fit(X_train, y_train)

#model accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
mat = confusion_matrix(y_test, y_pred)

print(mixture_model.lambda_m)
print(train_acc)
print(test_acc)
print(mat)

importances = pd.DataFrame({
	'feature':model_cols,
	'importance': np.round(model.feature_importances_,4)})
importances = importances.sort_values('importance', ascending = False)
print(importances)

#expected points
y_proba = model.predict_proba(X_test)
exp_points = np.dot(y_proba, mixture_model.lambda_m.T)

test_result_data['exp_points'] = exp_points

print(X_test.shape)
print(test_result_data.shape)

#test_result_data.to_csv('results.csv', index = False)



















