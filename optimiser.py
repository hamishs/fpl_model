import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import operator
import pulp

'''
altered form of: https://github.com/Torvaney/fpl-optimiser

'''


def add_position_dummy(df):
	'''one hot encode the positions into the df given '''
	for p in df.position.unique():
		df['is_' + str(p).lower()] = np.where(df.position == p, int(1), int(0))
	return df

def add_team_dummy(df):
	'''one hot encode the teams into the df given '''
	for t in df.team_id.unique():
		df['team_' + str(t).lower()] = np.where(df.team_id == t, int(1), int(0))
	return df

def get_optimal_squad(data, formation = '2-5-5-3', budget = 100.0, season = '2019/20',
	optimise_on='total_points', monitor = 'total_points'):
	''' formtion = None gives optimal 11 man squad'''
	
	min_player_cost = 4.0
	
	if formation is not None:
		n_players = sum(int(i) for i in formation.split('-'))
	else:
		n_players = 11

	season_stats = (
		data
		.loc[data.season_name == season]
		.reset_index()
		.assign(cost=lambda df: (df.value / 10.0))
		.pipe(add_position_dummy)
		.pipe(add_team_dummy)
	)

	players = season_stats.name

	fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)

	#dictionary of pulp variables with keys from names
	x = pulp.LpVariable.dict('x_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)

	#player score data
	player_points = dict(zip(season_stats.name, np.array(season_stats[optimise_on])))

	#objective function
	fpl_problem += sum([player_points[i] * x[i] for i in players])

	#constraints
	#position constraints
	position_names = ['gk', 'def', 'mid', 'fwd']
	
	player_position = dict(zip(season_stats.name, season_stats.position))
	player_gk = dict(zip(season_stats.name, season_stats.is_goalkeeper))
	player_def = dict(zip(season_stats.name, season_stats.is_defender))
	player_mid = dict(zip(season_stats.name, season_stats.is_midfielder))
	player_fwd = dict(zip(season_stats.name, season_stats.is_forward))
	player_monitor = dict(zip(season_stats.name, season_stats[monitor])) #store monitored statistic

	if formation is not None:
		position_constraints = [int(i) for i in formation.split('-')]
		constraints = dict(zip(position_names, position_constraints))
		
		fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= float(constraints['gk'])
		fpl_problem += sum([player_def[i] * x[i] for i in players]) <= float(constraints['def'])
		fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= float(constraints['mid'])
		fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= float(constraints['fwd'])
	else:
		position_constraints_upper = [int(i) for i in '1-5-5-3'.split('-')]
		position_constraints_lower = [int(i) for i in '1-3-3-1'.split('-')]
		positions_upper = ['gk_up', 'def_up', 'mid_up', 'fwd_up']
		positions_lower = ['gk_lo', 'def_lo', 'mid_lo', 'fwd_lo']
		constraints = dict(zip(positions_upper, position_constraints_upper))
		constraints.update(dict(zip(positions_lower, position_constraints_lower)))

		fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= float(constraints['gk_up'])
		fpl_problem += sum([player_def[i] * x[i] for i in players]) <= float(constraints['def_up'])
		fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= float(constraints['mid_up'])
		fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= float(constraints['fwd_up'])

		fpl_problem += sum([player_gk[i] * x[i] for i in players]) >= float(constraints['gk_lo'])
		fpl_problem += sum([player_def[i] * x[i] for i in players]) >= float(constraints['def_lo'])
		fpl_problem += sum([player_mid[i] * x[i] for i in players]) >= float(constraints['mid_lo'])
		fpl_problem += sum([player_fwd[i] * x[i] for i in players]) >= float(constraints['fwd_lo'])

		#must have 11 players
		fpl_problem += sum([x[i] for i in players]) == float(n_players)

	#budget constraints
	constraints['total_cost'] = budget
	player_cost = dict(zip(season_stats.name, season_stats.cost))
	fpl_problem += sum([player_cost[i] * x[i] for i in players]) <= float(constraints['total_cost'])

	#team constraints
	constraints['team'] = 3
	for t in season_stats.team_id:
		player_team = dict(zip(season_stats.name, season_stats['team_' + str(t)]))
		fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']

	#solve problem
	#(hiding print)
	fpl_problem.solve(solver = pulp.PULP_CBC_CMD(msg = 0))

	if formation is not None:
		chosen_formation = formation
	else:
		gks = int(sum([player_gk[i] * x[i].value() for i in players]))
		defs = int(sum([player_def[i] * x[i].value() for i in players]))
		mids = int(sum([player_mid[i] * x[i].value() for i in players]))
		fwds = int(sum([player_fwd[i] * x[i].value() for i in players]))
		chosen_formation = str(gks)+'-'+str(defs)+'-'+str(mids)+'-'+str(fwds)

	total_points = 0.0
	total_cost = 0.0
	total_monitor = 0.0
	optimal_squad = []
	for p in players:
		if x[p].value() != 0:
			total_points += player_points[p]
			total_cost += player_cost[p]
			total_monitor += player_monitor[p]

			optimal_squad.append({
				'name': p,
				'position': player_position[p],
				'cost': player_cost[p],
				optimise_on: player_points[p],
				monitor: player_monitor[p]
				})

	solution_info = {
		'formation': chosen_formation,
		optimise_on: total_points,
		'total_cost': total_cost,
		monitor: total_monitor
	}

	return optimal_squad, solution_info

def get_optimal_squad_w_subs(data, sub_weights = [1, 1, 1, 1], budget = 100.0, season = '2019/20',
	optimise_on='total_points', monitor = 'total_points'):
	'''
	optimises the squad given 15 players with 4 subs dampened by the weights
	weight 0 is gk, weights 1,2,3 are subs 1,2,3
	'''
	
	min_player_cost = 4.0

	season_stats = (
		data
		.loc[data.season_name == season]
		.reset_index()
		.assign(cost=lambda df: (df.value / 10.0))
		.pipe(add_position_dummy)
		.pipe(add_team_dummy)
	)

	players = season_stats.name

	fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)

	# x is starting line up, zi are subs
	#dictionary of pulp variables with keys from names
	x = pulp.LpVariable.dict('x_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)

	#sub variables
	z0 = pulp.LpVariable.dict('z0_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)
	z1 = pulp.LpVariable.dict('z1_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)
	z2 = pulp.LpVariable.dict('z2_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)
	z3 = pulp.LpVariable.dict('z3_ % s', players, lowBound = 0, upBound = 1,
		cat = pulp.LpInteger)

	#player score data
	player_points = dict(zip(season_stats.name, np.array(season_stats[optimise_on])))

	#objective function
	fpl_problem += sum([player_points[i] * (x[i] + z0[i] * sub_weights[0] + z1[i] * sub_weights[1] + z2[i] * sub_weights[2] + z3[i] * sub_weights[3]) for i in players])

	#constraints
	constraints = {}
	#position constraints
	position_names = ['gk', 'def', 'mid', 'fwd']
	
	player_position = dict(zip(season_stats.name, season_stats.position))
	player_gk = dict(zip(season_stats.name, season_stats.is_goalkeeper))
	player_def = dict(zip(season_stats.name, season_stats.is_defender))
	player_mid = dict(zip(season_stats.name, season_stats.is_midfielder))
	player_fwd = dict(zip(season_stats.name, season_stats.is_forward))
	player_monitor = dict(zip(season_stats.name, season_stats[monitor])) #store monitored statistic

	#starting lineup constraints
	position_constraints_upper = [1,5,5,3]
	position_constraints_lower = [1,3,3,1]
	constraints_upper = dict(zip(position_names, position_constraints_upper))
	constraints_lower = dict(zip(position_names, position_constraints_lower))

	fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= float(constraints_upper['gk'])
	fpl_problem += sum([player_def[i] * x[i] for i in players]) <= float(constraints_upper['def'])
	fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= float(constraints_upper['mid'])
	fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= float(constraints_upper['fwd'])

	fpl_problem += sum([player_gk[i] * x[i] for i in players]) >= float(constraints_lower['gk'])
	fpl_problem += sum([player_def[i] * x[i] for i in players]) >= float(constraints_lower['def'])
	fpl_problem += sum([player_mid[i] * x[i] for i in players]) >= float(constraints_lower['mid'])
	fpl_problem += sum([player_fwd[i] * x[i] for i in players]) >= float(constraints_lower['fwd'])

	fpl_problem += sum([x[i] for i in players]) == 11 #starting lineup

	fpl_problem += sum([z0[i] for i in players]) == 1 #gk sub
	fpl_problem += sum([player_gk[i] * z0[i] for i in players]) == 1

	fpl_problem += sum([z1[i] for i in players]) == 1 #sub 1
	fpl_problem += sum([z2[i] for i in players]) == 1 #sub 2
	fpl_problem += sum([z3[i] for i in players]) == 1 #sub 3

	fpl_problem += sum([player_def[i] * (x[i] + z1[i] + z2[i] + z3[i]) for i in players]) == 5
	fpl_problem += sum([player_mid[i] * (x[i] + z1[i] + z2[i] + z3[i]) for i in players]) == 5
	fpl_problem += sum([player_fwd[i] * (x[i] + z1[i] + z2[i] + z3[i]) for i in players]) == 3

	#no duplicates constraint:
	for i in players:
		fpl_problem += (x[i] + z0[i] + z1[i] + z2[i] + z3[i]) <= 1

	#budget constraints
	constraints['total_cost'] = budget
	player_cost = dict(zip(season_stats.name, season_stats.cost))
	fpl_problem += sum([player_cost[i] * (x[i] + z0[i] + z1[i] + z2[i] + z3[i]) for i in players]) <= float(constraints['total_cost'])

	#team constraints
	constraints['team'] = 3
	for t in season_stats.team_id:
		player_team = dict(zip(season_stats.name, season_stats['team_' + str(t)]))
		fpl_problem += sum([player_team[i] * (x[i] + z0[i] + z1[i] + z2[i] + z3[i]) for i in players]) <= constraints['team']

	#solve problem
	#(hiding print)
	fpl_problem.solve(solver = pulp.PULP_CBC_CMD(msg = 0))

	''' amend to starting lineup'''
	gks = int(sum([player_gk[i] * x[i].value() for i in players]))
	defs = int(sum([player_def[i] * x[i].value() for i in players]))
	mids = int(sum([player_mid[i] * x[i].value() for i in players]))
	fwds = int(sum([player_fwd[i] * x[i].value() for i in players]))
	chosen_formation = str(gks)+'-'+str(defs)+'-'+str(mids)+'-'+str(fwds)

	total_points = 0.0
	total_cost = 0.0
	total_monitor = 0.0
	optimal_squad = []
	for p in players:
		if x[p].value() != 0:
			total_points += player_points[p]
			total_cost += player_cost[p]
			total_monitor += player_monitor[p]

			optimal_squad.append({
				'name': p,
				'position': player_position[p],
				'cost': player_cost[p],
				optimise_on: player_points[p],
				monitor: player_monitor[p],
				'sub': False
				})

		if z0[p].value()+z1[p].value()+z2[p].value()+z3[p].value() != 0:
			total_cost += player_cost[p]
			if z0[p].value() != 0:
				sub = 0
			elif z1[p].value() != 0:
				sub = 1
			elif z2[p].value() != 0:
				sub = 2
			elif z3[p].value() != 0:
				sub = 3
			optimal_squad.append({
				'name': p,
				'position': player_position[p],
				'cost': player_cost[p],
				optimise_on: player_points[p],
				monitor: player_monitor[p],
				'sub': sub
				})

	solution_info = {
		'formation': chosen_formation,
		optimise_on: total_points,
		'total_cost': total_cost,
		monitor: total_monitor
	}

	return optimal_squad, solution_info

def optimal_transfer_plan(data, current_squad, formation = '2-5-5-3', free_transfers = 1,
	budget = 100.0, season = '2019/20', optimise_on='total_points', monitor = 'total_points'):
	''' optimal squad including -4 per transfer '''
	
	'''current squad as dict'''
	y = current_squad

	min_player_cost = 4.0
	
	if formation is not None:
		n_players = sum(int(i) for i in formation.split('-'))
	else:
		n_players = 11

	season_stats = (
		data
		.loc[data.season_name == season]
		.reset_index()
		.assign(cost=lambda df: (df.value / 10.0))
		.pipe(add_position_dummy)
		.pipe(add_team_dummy)
	)

	players = season_stats.name

	#error when players are new in gw
	y = dict.fromkeys(players,0)
	for p in players:
		try: #may error if p not in last gw so not in keys
			if current_squad[p] == 1: 
				y[p] = 1
		except KeyError:
				y[p] = 0

	missing_players = sum([current_squad[p] for p in current_squad.keys()])-sum([y[p] for p in y.keys()])
	if missing_players != 0:	
		print('missing players')
		for p in current_squad.keys():
			if p not in y.keys():
				print(p)

	solutions = {}

	for ft in range(free_transfers+1):
		print('Try {} free transfers'.format(ft))

		fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)

		#dictionary of pulp variables with keys from names
		x = pulp.LpVariable.dict('x_ % s', players, lowBound = 0, upBound = 1,
			cat = pulp.LpInteger)

		#player score data
		player_points = dict(zip(season_stats.name, np.array(season_stats[optimise_on])))

		#objective function
		fpl_problem += sum([(player_points[i] - 4 * (1-y[i])) * x[i] for i in players]) + 4 * ft
		
		#constraints
		#position constraints
		position_names = ['gk', 'def', 'mid', 'fwd']
		
		player_position = dict(zip(season_stats.name, season_stats.position))
		player_gk = dict(zip(season_stats.name, season_stats.is_goalkeeper))
		player_def = dict(zip(season_stats.name, season_stats.is_defender))
		player_mid = dict(zip(season_stats.name, season_stats.is_midfielder))
		player_fwd = dict(zip(season_stats.name, season_stats.is_forward))
		player_monitor = dict(zip(season_stats.name, season_stats[monitor])) #store monitored statistic

		if formation is not None:
			position_constraints = [int(i) for i in formation.split('-')]
			constraints = dict(zip(position_names, position_constraints))
			
			fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= float(constraints['gk'])
			fpl_problem += sum([player_def[i] * x[i] for i in players]) <= float(constraints['def'])
			fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= float(constraints['mid'])
			fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= float(constraints['fwd'])
		else:
			position_constraints_upper = [int(i) for i in '1-5-5-3'.split('-')]
			position_constraints_lower = [int(i) for i in '1-3-3-1'.split('-')]
			positions_upper = ['gk_up', 'def_up', 'mid_up', 'fwd_up']
			positions_lower = ['gk_lo', 'def_lo', 'mid_lo', 'fwd_lo']
			constraints = dict(zip(positions_upper, position_constraints_upper))
			constraints.update(dict(zip(positions_lower, position_constraints_lower)))

			fpl_problem += sum([player_gk[i] * x[i] for i in players]) <= float(constraints['gk_up'])
			fpl_problem += sum([player_def[i] * x[i] for i in players]) <= float(constraints['def_up'])
			fpl_problem += sum([player_mid[i] * x[i] for i in players]) <= float(constraints['mid_up'])
			fpl_problem += sum([player_fwd[i] * x[i] for i in players]) <= float(constraints['fwd_up'])

			fpl_problem += sum([player_gk[i] * x[i] for i in players]) >= float(constraints['gk_lo'])
			fpl_problem += sum([player_def[i] * x[i] for i in players]) >= float(constraints['def_lo'])
			fpl_problem += sum([player_mid[i] * x[i] for i in players]) >= float(constraints['mid_lo'])
			fpl_problem += sum([player_fwd[i] * x[i] for i in players]) >= float(constraints['fwd_lo'])

			#must have 11 players
			fpl_problem += sum([x[i] for i in players]) == float(n_players)

		#add transfer constraint
		fpl_problem += sum(x[i] * (1 - y[i]) for i in players) >= float(ft)

		#feasible transfers constraint
		#positions of transferred in players must be the same as transferred out
		#transferred in[i]  = x[i] * (1 - y[i])
		#transferred out[i] = y[i] * (1 - x[i])
		fpl_problem += sum([player_gk[i] * x[i] * (1 - y[i]) for i in players]) - sum([player_gk[i] * y[i] * (1 - x[i]) for i in players]) == 0
		fpl_problem += sum([player_def[i] * x[i] * (1 - y[i]) for i in players]) - sum([player_def[i] * y[i] * (1 - x[i]) for i in players]) == 0
		fpl_problem += sum([player_mid[i] * x[i] * (1 - y[i]) for i in players]) - sum([player_mid[i] * y[i] * (1 - x[i]) for i in players]) == 0
		fpl_problem += sum([player_fwd[i] * x[i] * (1 - y[i]) for i in players]) - sum([player_fwd[i] * y[i] * (1 - x[i]) for i in players]) == 0

		#budget constraints
		constraints['total_cost'] = budget
		player_cost = dict(zip(season_stats.name, season_stats.cost))
		fpl_problem += sum([player_cost[i] * x[i] for i in players]) <= float(constraints['total_cost'])

		#team constraints
		constraints['team'] = 3
		for t in season_stats.team_id:
			player_team = dict(zip(season_stats.name, season_stats['team_' + str(t)]))
			fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']

		#solve problem
		#(hiding print)
		fpl_problem.solve(solver = pulp.PULP_CBC_CMD(msg = 0))

		if formation is not None:
			chosen_formation = formation
		else:
			gks = int(sum([player_gk[i] * x[i].value() for i in players]))
			defs = int(sum([player_def[i] * x[i].value() for i in players]))
			mids = int(sum([player_mid[i] * x[i].value() for i in players]))
			fwds = int(sum([player_fwd[i] * x[i].value() for i in players]))
			chosen_formation = str(gks)+'-'+str(defs)+'-'+str(mids)+'-'+str(fwds)

		total_points = 0.0
		total_cost = 0.0
		total_monitor = 0.0
		transfers = 0
		transfer_cost = 4 * ft
		optimal_squad = []
		t_out = []
		t_in = []
		for p in players:
			if x[p].value() != 0:
				total_points += player_points[p]
				total_cost += player_cost[p]
				total_monitor += player_monitor[p]
				transfers += (1-y[p])
				transfer_cost += -4*(1-y[p])

				if y[p] == 0:
					t_in.append(p)

				optimal_squad.append({
					'name': p,
					'position': player_position[p],
					'cost': player_cost[p],
					optimise_on: player_points[p],
					monitor: player_monitor[p]
					})

			if y[p] != 0:
				if x[p].value() == 0:
					t_out.append(p)

		solution_info = {
			'formation': chosen_formation,
			optimise_on: total_points,
			'total_cost': total_cost,
			monitor: total_monitor,
			'transfers': transfers,
			'transfer_cost': transfer_cost,
			'total_predicted': total_points + transfer_cost,
			'out' : t_out,
			'in' : t_in
		}

		solutions[ft] = (optimal_squad, solution_info)

	opt_ft = max(solutions.keys(), key = (lambda key: solutions[key][1]['total_predicted']))

	(optimal_squad, solution_info) = solutions[opt_ft]

	return optimal_squad, solution_info



