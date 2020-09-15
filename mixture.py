import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from scipy.stats import poisson
from sklearn.cluster import KMeans

class PoissonMixtureModel():

	'''
	Poisson mixture model with K poisson variables.
	Fitted with expectation maximisation algorithm.
	'''

	def __init__(self, y = None, K = 5, n_iter = 1000, max_tol = 0.00001, msg = True):
		''''''
		self.K = K

		if y is not None:
			self.fit(y, n_iter = n_iter, max_tol = max_tol, msg = msg)

	def fit(self, y, n_iter = 1000, max_tol = 0.00001, msg = True):
		'''
		Fit a poisson mixture model to y with k poisson hidden variables
		'''
		y = np.maximum(y,0)
		N = y.shape[0]

		#inital guesses
		kmeans = KMeans(n_clusters = self.K)
		kmeans.fit(y.reshape(-1,1))
		clusters = kmeans.predict(y.reshape(-1,1))
		pi_init = np.array([np.sum(np.where(clusters == x, 1, 0)) for x in range(self.K)]) / N
		lambda_init = np.random.rand(self.K) * 10

		#EM algorithm
		n_iter = 1000
		max_tol = 0.00001

		#responsibilites
		r_im = np.array([poisson.pmf(y, lmda) for lmda in lambda_init]).T * pi_init
		r_im = r_im / r_im.sum(axis = 1, keepdims = True)

		#params
		lambda_m = np.dot(r_im.T, y) / r_im.sum(axis = 0)
		pi_m = r_im.sum(axis = 0) / N

		#log likelihood
		l = np.log((np.array([poisson.pmf(y, lmda) for lmda in lambda_init]).T * pi_init).sum(axis = 1)).sum()

		likelihoods = [l]
		for i in range(n_iter):
			#E step
			'''
			Current params Theta. Compute responsibilities for
			all data points and mixture components (NxK matrix)
			with sum of all rows = 1

			Eval responsibilities:
				r_im = P_m(x)pi_m / sum_k(P_k(x)pi_k)
			'''

			r_im = np.array([poisson.pmf(y, lmda) for lmda in lambda_m]).T * pi_m
			r_im = r_im / r_im.sum(axis = 1, keepdims = True)

			#M step
			'''
			Calculate new parameter values with responsibilities.

			Update params:
				lambda_m = sum_i(r_im*x_i) / sum_i(r_im)
				pi_m = sum_i(r_im) / N
			'''

			lambda_m = np.dot(r_im.T, y) / r_im.sum(axis = 0)
			pi_m = r_im.sum(axis = 0) / N

			#compute log likelihood
			l = np.log((np.array([poisson.pmf(y, lmda) for lmda in lambda_m]).T * pi_m).sum(axis = 1)).sum()

			#tol
			epsilon = l - likelihoods[-1]
			if abs(epsilon) < max_tol:
				if msg:
					print('Max tol achieved after {} iterations'.format(i))
					print('Loglikelihood: {}'.format(l))
				break
			
			likelihoods.append(l)
			
			if i == n_iter-1:
				if msg:
					print('Max iter reached')
					print('Loglikelihood: {}'.format(l))

			self.lambda_m = lambda_m
			self.pi_m = pi_m
			
	
	def responsibilities(self, y):
		'''find the responsibilities for y'''
		y = np.maximum(y,0)
		r_im = np.array([poisson.pmf(y, lmda) for lmda in self.lambda_m]).T * self.pi_m
		r_im = r_im / r_im.sum(axis = 1, keepdims = True)
		return r_im


	def cluster(self, y):
		'''get the cluster for each value of y'''
		y = np.maximum(y,0)
		r_im = self.responsibilities(y)
		return np.argmax(r_im, axis = 1)


if __name__ == '__main__':
	filename = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2019-20/gws/merged_gw.csv'
	df = pd.read_csv(filename)
	y = df['total_points'].to_numpy()

	mixture = PoissonMixtureModel(K = 4)
	mixture.fit(y)


















