import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensorflow.python import debug as tf_debug

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

import pandas.io.common

import scipy.optimize as opt
import os

from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import minimize

class RLogRegression():

	def __init__(self, path=None):

		try:
			self._df = pd.read_csv(path, header=None, delimiter=',')
		except pandas.io.common:
			print("Error! Wrong data| file not exist!")

		self._l = len(self._df[0])
		self.x = self._df.iloc[:, 0:2]
		self.y = self._df[2]

		self._poly = PolynomialFeatures(6)
		self.features = self._poly.fit_transform(self.x)

	def sygmoid(self, data = None):

		if (data is None):
			data = self.x
		g_z = 1 / (1 + np.exp(-data))
		return (g_z)

	def plotData2(self, label_x, label_y, label_pos, label_neg, axes=None):

		neg = self._df.iloc[:,2] == 0
		pos = self._df.iloc[:,2] == 1

		if axes == None:
			axes = plt.gca()

		axes.scatter(self._df[pos][0], self._df[pos][1], marker='+', color='k', s=60, linewidths=2, label=label_pos)
		axes.scatter(self._df[neg][0], self._df[neg][1], marker='o', color='y', label=label_neg)
		axes.set_xlabel(label_x)
		axes.set_ylabel(label_y)
		axes.legend(frameon=True, fancybox=True)

		plt.show()

	def costFunction(self, thetta, X, y, lambd):

		hypotesis = self.sygmoid(np.dot(thetta.T, self.features.T))
		# print(hypotesis)
		lg1 = np.log(hypotesis)
		lg2 = np.log(1 - hypotesis)
		first_part = -self.y * lg1
		second_part = (1 - self.y) * lg2
		res = first_part - second_part
		first_sum = np.sum(res) / self._l

		second_sum = np.sum(np.power(thetta, 2)) * (lambd / (2 * self._l))

		answer = first_sum + second_sum
		return (answer)

	def gradient(self, thetta, X, y, lambd): 

		hypotesis = self.sygmoid(np.dot(thetta, X.T))
		t = np.subtract(hypotesis, y)
		summ = np.dot(t, X)
		first_part = np.dot(1 / self._l, summ)

		len_t = len(thetta)
		for i in range(1, len_t):

			first_part[i] += (lambd / self._l) * thetta[i]

		return (first_part)

	def optimizeFunc(self, thetta=np.zeros(28), lambd=1):
			
		result = opt.fmin_tnc(func=self.costFunction, x0=thetta, fprime=self.gradient, args=(self.features, self.y, lambd))
		cs = self.costFunction(result[0], self.x, self.y, lambd)
		print(result)
		return (result[0])

	def predict(self, thetta, X, threshold=0.5):

		prediction = self.sygmoid(np.dot(thetta, X.T))
		return (prediction >= threshold)

	def plotData(self, label_x, label_y, label_pos, label_neg, axes=None):

		neg = self._df.iloc[:,2] == 0
		pos = self._df.iloc[:,2] == 1

		if axes == None:
			axes = plt.gca()

		axes.scatter(self._df[pos][0], self._df[pos][1], marker='+', c='k', s=60, linewidths=2, label=label_pos)
		axes.scatter(self._df[neg][0], self._df[neg][1], marker='o', c='y', label=label_neg)
		axes.set_xlabel(label_x)
		axes.set_ylabel(label_y)
		axes.legend(frameon=True, fancybox=True)


	def plot(self, thetta=np.zeros(28), lambd=1):

		fig, axes = plt.subplots(1, 3, sharey = True, figsize=(17, 5))

		for i, C in enumerate([0, 1, 100]):

			lambd = C
			res = minimize(self.costFunction, thetta, args=(self.features, self.y, lambd), method=None, jac=self.gradient, options={'maxiter':3000})

			accurancy = 100 * np.sum(self.predict(res.x, self.features) == self.y) / self._l

			self.plotData("Microchip Test 1", "Microchip Test 2", "y = 1", "y = 0", axes.flatten()[i])

			x1_min, x1_max = self.x.iloc[:,0].min(), self.x.iloc[:,0].max()
			x2_min, x2_max = self.x.iloc[:,1].min(), self.x.iloc[:,1].max()

			print(x1_min, x1_max)

			xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
			h = self.sygmoid(self._poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
			h = h.reshape(xx1.shape)
			axes.flatten()[i].contour(xx1, xx2, h, levels=0.5, linewidths=1, colors='g')
			axes.flatten()[i].set_title("Train accurancy {}% with lambda = {}".format(np.around(accurancy, decimals=2), C))
			#plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

		plt.show()
		print(accurancy)


def main():

	path = os.getcwd() + '/ex2data2.txt'
	RLR = RLogRegression(path)
	# lmbd = 1
	# thetta = np.zeros(28)
	lmbd = 10
	thetta = np.ones(28)
	#RLR.plotData('Microchip test 1', 'Microchip test 2', 'y=1', 'y=0')
	# print(RLR.costFunction(thetta, RLR.features, RLR.y, lmbd))
	# print(RLR.gradient(thetta, RLR.features, RLR.y, lmbd))
	# print(RLR.optimsizeFunc(thetta, lmbd))
	# RLR.optimizeFunc(thetta, lmbd)
	RLR.plot()


if __name__ == "__main__":
	main()
