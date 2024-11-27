import numpy as np
import pandas as pd

class dataGenerator():

	def __init__(self):
		self.data = pd.DataFrame(columns=['x','y'])
		input, output = self.generateOutput(2,1,10)
		self.data['x'] = input.tolist()
		self.data['y'] = output.tolist()

	def generateOutput(self,scale,bias,quantity):
		input = 100*(np.random.rand(1000))
		output = scale*input+bias+(input*np.random.rand())
		return [input, output]

if __name__ == "__main__":
	a = np.random.rand()
	generator = dataGenerator()
	generator.data.to_pickle('./data/pickles/linearData.pkl')


