import torch as tt
from network import Model
from time import time
import json
from criterion import Criterion
from data import Dataset
from data import DataManager
from data import DataAugmenter
import numpy as np
import matplotlib.pyplot as plt
from network import Model
from visual import Visualiser
from os import system



class Learner():
	def __init__(self, dataloader):
		self.configuration = json.load(open("./configuration.json", "r"))

		self.dataloader = dataloader
		self.criterion = Criterion()
		self.theta_estimate = None
		self.transform_estimate = None

		self.device = self.configuration["device"]
		self.model = Model().to(self.device)
		self.optimiser = tt.optim.AdamW(
			self.model.parameters(),
            lr=self.configuration["learning_rate_start"],
        )
		# self.optimiser = tt.optim.SGD(
		# 	self.model.parameters(),
        #     lr=self.configuration["learning_rate"],
        # )

	def set_lr(self, learning_rate):
		for param_group in self.optimiser.param_groups:
			param_group['lr'] = learning_rate

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
		for batch in self.dataloader[phase]:
			batch_images = batch["image"].to(self.device)

			# Get estimated transform
			theta_estimate, self.transform_estimate = self.model(batch_images)
			self.theta_estimate = dict(
				theta_1=theta_estimate[:,0],
				theta_2=theta_estimate[:,1],
				theta_3=theta_estimate[:,2],
				theta_4=theta_estimate[:,3],
				theta_5=theta_estimate[:,4],
				theta_6=theta_estimate[:,5],
				)
			batch_transform = batch['transform'].to(self.device)

			batch_theta = dict(
				theta_1=batch_transform[:,0,0],
				theta_2=batch_transform[:,0,1],
				theta_3=batch_transform[:,0,2],
				theta_4=batch_transform[:,1,0],
				theta_5=batch_transform[:,1,1],
				theta_6=batch_transform[:,1,2],
				)

			# Loss on estimate
			loss = Criterion.forward(self, estimate=self.theta_estimate, reference=batch_theta)
			loss_step += loss.item() # cast to float

			if phase == "train":				
				loss.backward() # Required by optimiser for loss-gradient

				# Update model
				self.optimiser.step()
				self.optimiser.zero_grad()

			# accuracy[0] += tt.sum(tt.argmax(self.estimate, dim=1) == tt.argmax(rotation, dim=1)).item()
			# accuracy[1] += len(rotation)

		# Log step
		loss_step = loss_step / len(self.dataloader[phase])
		log[f"loss_{phase}"].append(loss_step)
		# log[f"accuracy_{phase}"].append(accuracy[0] / accuracy[1])
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
		data_loader_train = tt.utils.data.DataLoader(data_set_train, batch_size=config['batch_size'], shuffle=True,num_workers=4) #Example
		data_loader_val = tt.utils.data.DataLoader(data_set_val, batch_size=config['batch_size'], shuffle=True,num_workers=4) #Example
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

		fig_loss, ax_loss = plt.subplots()  # For loss plot
		fig_images, axes = plt.subplots(1, 3, figsize=(6, 3))

		learning_rate_schedule = np.linspace(
			learner.configuration['learning_rate_start'],
			learner.configuration['learning_rate_final'],
			learner.configuration['epochs'])

		# define eval step
		for epoch in range(learner.configuration["epochs"]):

			# Set Scheduled learning rate
			learner.set_lr(learning_rate_schedule[epoch])
			print(learning_rate_schedule[epoch])
			# print(f"Epoch: {epoch}")
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
				f"[Epoch {epoch}: {(log['time_train'][-1] + log['time_val'][-1]):.2f}s] Train loss: {log['loss_train'][-1]:.5f}, Val loss: {log['loss_val'][-1]:.5f}",
				end="",
			)
			if log["saved_weights"][-1]:
				print(" [saved]")
			else:
				print("")

			# log metadata
			json.dump(log, open(path_logs/f"test_log_k_split_{k_split_nr}.json", "w"), sort_keys=True, indent=4)

			# plot progress
			
			ax_loss.clear()  # clear previous lines each epoch
			ax_loss.plot(range(epoch + 1), log["loss_train"], label="train_loss")
			ax_loss.plot(range(epoch + 1), log["loss_val"], label="val_loss")
			ax_loss.set_xlabel("Epoch")
			ax_loss.set_ylabel("Loss")
			ax_loss.set_title(f"Training: {run} split_{k_split_nr}")
			ax_loss.legend(loc=1)

			fig_loss.savefig(path_plots / f"test_k_split_{k_split_nr}.png")

			batch = next(iter(learner.dataloader['val']))
			batch_images = batch['image']
			
			# Get Transforms
			batch_transform = batch['transform'][0]
			batch_transform_inverse = data_set_val.augmenter.invertTransform(batch_transform)
			image_corrected = data_set_val.augmenter.applyTransform(batch_images[0],batch_transform_inverse)

			estimate_transform_inverse = data_set_val.augmenter.invertTransform(learner.transform_estimate[0].to('cpu'))
			image_estimated = data_set_val.augmenter.applyTransform(batch_images[0],estimate_transform_inverse)
			print("\n")
			
			axes[0].imshow(batch_images[0,0,:,:], cmap="gray")
			axes[0].set_title(f"Batch Image")
			axes[0].axis("off")

			axes[1].imshow(image_estimated[0,:,:], cmap="gray")
			axes[1].set_title(f"Estimated Correction")
			axes[1].axis("off")

			axes[2].imshow(image_corrected[0,:,:], cmap="gray")
			axes[2].set_title(f"Real Correction")
			axes[2].axis("off")
			fig_images.savefig("output/plots/rotations.png", dpi=300, bbox_inches="tight")


	print(f"[Training {run}] done")
	system('afplay data/finish-notification.wav') 