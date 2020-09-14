# fpl model

Predictive model and team selecter for the popular fantasy football game:
Fantasy Premier League (FPL). 

Points are clustered using a poisson mixutre model trained using expectation maximisation. Then a gradient boosting classifier is trained on 3 seasons (2016/17-2018/19) of FPL data and xG/xA data from understat. The 2019/20 season is reserved for testing. The model takes input of the players results in the last gameweek (gw) as well as short and long term averages of each feature over the gw history. It also uses static data such as the player's points total in previous seasons and next gw info e.g. fixture difficulty and home/away. It outputs a prediction of the points in the next GW.

For a given GW the predicted points are calculated then an optimiser selects the team given budget, team and position constraints. The optimiser is built from an optimiser developed by Torvaney (link below). It is expanded to allow the algorithm to select the formation, encode transfer costs and choose subs.

Links
FPL data: https://github.com/vaastav/Fantasy-Premier-League
Understat API: https://understat.readthedocs.io/en/latest/
Optimiser: https://github.com/Torvaney/fpl-optimiser