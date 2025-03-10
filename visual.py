import matplotlib.pyplot as plt
import numpy as np

class Visualiser():
	def __init__(self):
		return
	
	def compareAngles(self, *lines):
		fig = plt.figure()
		for line in lines:
			x = np.linspace(0,len(line)-1,len(line))
			plt.plot(x, line)
		return fig
	
if __name__ == "__main__":
	vis = Visualiser()
	refs = [1, 4, 3]
	ests = [2,3, 4]
	fig = vis.compareAngles(refs, ests)
	# plt.show(fig)
	fig.savefig("test-delete-me.png")
