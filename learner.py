import torch as tt
from network import Model
from time import time
import json
from criterion import Criterion
from data import Dataset
from data import DataManager
from data import dataAugmenter
import numpy as np
import matplotlib.pyplot as plt
from network import Model
from visual import Visualiser


class Learner():
	def __init__(self, dataloader):
		self.configuration = json.load(open("./configuration.json", "r"))

		self.dataloader = dataloader
		self.criterion = Criterion()
		self.estimate = None

		self.device = self.configuration["device"]
		self.model = Model().to(self.device)
		self.optimiser = tt.optim.AdamW(
			self.model.parameters(),
            lr=self.configuration["learning_rate"],
        )

	def step(self, phase):
		''' Determine Training/Testing/Validating'''
		if phase == "train":
			self.model.train()
			tt.set_grad_enabled(True)
		else:
			self.model.eval()
			tt.set_grad_enabled(False)

		''' Initialse Loss and Time'''
		loss_step = 0.0
		time_step = time()

		'''Iterate over Dataloader'''
		accuracy = [0, 0]

		# counter = 1
		for data in self.dataloader[phase]:
			# if counter%100==0:
			# 	print(f"Step: {counter}")
			# counter += 1
			image = data["image"].to(self.device)
			label = data["label"].to(self.device)

			# Get estimated label
			self.estimate = self.model(image)
			rotation = next(iter(self.dataloader[phase]))['rotation'].to(tt.float32).to(self.device).view(self.configuration['batch_size'],1)
			# Loss on estimate
			loss = Criterion.forward(self, prediction=self.estimate, reference=rotation)
			loss_step += loss.item() # cast to float

			if phase == "train":				
				loss.backward() # Required by optimiser for loss-gradient

				# Update model
				self.optimiser.step()
				self.optimiser.zero_grad()

			accuracy[0] += tt.sum(tt.argmax(self.estimate, dim=1) == tt.argmax(rotation, dim=1)).item()
			accuracy[1] += len(rotation)

		# Log step
		loss_step = loss_step / len(self.dataloader[phase])
		log[f"loss_{phase}"].append(loss_step)
		log[f"accuracy_{phase}"].append(accuracy[0] / accuracy[1])
		log[f"time_{phase}"].append(time() - time_step)

	def setDevice(self, device):
		self.device = device

	def setConfig(self,path_config):
		self.configuration = json.load(open(path_config, "r"))
		
	
if __name__ == "__main__":
	print('=== RAN AS FILE [learntern/learner.py] ===')

	'''Some Environment Variables'''
	# Set Output Paths
	manager = DataManager()
	here = manager.getParentDir()
	path_logs = here/"output"/"logs"
	path_weights = here/"output"/"weights"
	path_plots = here/"output"/"plots"

	# Import Data
	images = np.load(here/"data"/"arrays"/"images.npy")
	labels = np.load(here/"data"/"arrays"/"labels.npy")
	splits = np.load(here/"data"/"splits"/"5-fold-indices.npz")


	'''Important Variables'''
	# Name of this run
	run = "Test [Ran as File]"
	config = json.load(open("./configuration.json", "r"))
	used_splits = ['00']

	for k_split_nr in used_splits: #config.used_splits

		split_train = f"train_{k_split_nr}"
		split_val = f"val_{k_split_nr}"

		data_set_train = Dataset(images[splits[split_train]],labels[splits[split_train]])
		data_set_val = Dataset(images[splits[split_val]],labels[splits[split_val]])

		# Define data loader
		data_loader_train = tt.utils.data.DataLoader(data_set_train, batch_size=config['batch_size'], shuffle=True) #Example
		data_loader_val = tt.utils.data.DataLoader(data_set_val, batch_size=config['batch_size'], shuffle=True) #Example
		data_loader = dict(train=data_loader_train,val=data_loader_val)

		# Define model with dataloader
		learner = Learner(data_loader)

		# Create log variable for metadata
		log = dict(
				name="Test",
				epochs="",
				accuracy_train=[],
				accuracy_val=[],
				loss_train=[],
				loss_val=[],
				time_train=[],
				time_val=[],
				saved_weights=[],
				configuration=learner.configuration,
				device=learner.device,
			)

		# define eval step
		for epoch in range(learner.configuration["epochs"]):
			print(f"Epoch: {epoch}")
			# train
			learner.step("train")
			# evaluate
			learner.step("val")

			# Save weights if validation loss is lower than previous
			if epoch == 0 or log["loss_val"][-1] < min(log["loss_val"][:-1]):
				tt.save(learner.model.state_dict(), path_weights/f"weights_k_split_{k_split_nr}.pt")
				log["saved_weights"].append(True)
			else:
				log["saved_weights"].append(False)

			# print epoch overview
			print(
				f"[Epoch {epoch}: {(log['time_train'][-1] + log['time_val'][-1]):.2f}s] Train loss: {log['loss_train'][-1]:.5f}, Train acc.: {log['accuracy_train'][-1]:.5f}, Val loss: {log['loss_val'][-1]:.5f}, Val acc.: {log['accuracy_val'][-1]:.5f}",
				end="",
			)
			if log["saved_weights"][-1]:
				print(" [saved]")
			else:
				print("")

			# log metadata
			json.dump(log, open(path_logs/f"test_log_k_split_{k_split_nr}.json", "w"), sort_keys=True, indent=4)

			# plot progress
			plt.figure()
			plt.plot(range(epoch + 1), log["loss_train"], label="train_loss")
			plt.plot(range(epoch + 1), log["loss_val"], label="val_loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.title(f"Training: {run}"+f"split_{k_split_nr}")
			plt.legend(loc=1)
			plt.savefig(path_plots/f"test_k_split_{k_split_nr}.png")
			plt.close()

			batch = next(iter(learner.dataloader['val']))
			batch_images = batch['image']
			estimate_rotations = Model().forward(batch_images)

			batch_rotations = batch['rotation']
			augmenter = dataAugmenter()
			batch_images_straight = augmenter.rotateBatch(batch['image'], -batch_rotations)

			batch_images_estimated = dataAugmenter().rotateBatch(batch_images,-estimate_rotations)
			
			print(batch['rotation'])
			print(estimate_rotations)

			fig, axes = plt.subplots(1, 2, figsize=(6, 3))
			axes[0].imshow(batch_images[0,0,:,:], cmap="gray")
			axes[0].set_title(f"Batch Image 1")
			axes[0].axis("off")

			axes[1].imshow(batch_images_estimated[0,0,:,:], cmap="gray")
			axes[1].set_title(f"Batch Image 2")
			axes[1].axis("off")
			fig.savefig("output/plots/rotations.png", dpi=300, bbox_inches="tight")

			fig = Visualiser().compareAngles(batch_rotations,estimate_rotations)
			fig.savefig("output/plots/test.png")
	print(f"[Training {run}] done")