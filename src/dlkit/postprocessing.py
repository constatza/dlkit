import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_losses(train_losses, val_losses):
	"""Plot the training and validation losses."""
	fig, ax = plt.subplots()
	ax.plot(train_losses, label='Training loss')
	ax.plot(val_losses, label='Validation loss')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss')
	# set the scale to log
	ax.set_yscale('log')
	ax.legend()
	return fig


def save_model(model, path):
	"""Save the model to the path."""
	torch.save(model.state_dict(), path)


def load_model(model, path):
	"""Load the model from the path."""
	model.load_state_dict(torch.load(path))
	return model


def plot_parametric_predictions(
	y_pred,
	y_true,
	title=None,
	x_label=None,
	y_label=None,
	parameter=None,
	plotter=None,
	show=True,
	kwargs_true=None,
	kwargs_pred=None,
):
	"""Plot the predictions of the model."""
	# set plt style to scientific
	if isinstance(y_pred, torch.Tensor):
		y_pred = y_pred.detach().cpu().numpy().squeeze()
	if isinstance(y_true, torch.Tensor):
		y_true = y_true.detach().cpu().numpy().squeeze()

	if title is None:
		title = 'Model Fit'
	if parameter is None:
		parameter = np.arange(y_pred.shape[0])
	if x_label is None:
		x_label = 'N'
	if y_label is None:
		y_label = 'Output'
	if kwargs_true is None:
		kwargs_true = {'linestyle': '--'}
	if kwargs_pred is None:
		kwargs_pred = {'linestyle': '-'}

	with plt.style.context('science'):
		fig, ax = plt.subplots(figsize=(8, 5))
		if plotter is None:
			plotter = ax.plot

		plotter(parameter, y_true, label='True', **kwargs_true)
		plotter(parameter, y_pred, label='Predicted', **kwargs_pred)
		ax.set_title(title)
		ax.set_xlabel(x_label)
		ax.set_ylabel(y_label)
		ax.legend()
	if show:
		plt.show()
	return fig


def plot_pred_vs_true(y_pred, y_true, title=None, **kwargs):
	"""Plot predicted vs true values."""
	fig, ax = plt.subplots(figsize=(8, 5))
	# change marker to x shape
	ax.scatter(y_true, y_pred, marker='o', **kwargs)
	ax.set_xlabel('True')
	ax.set_ylabel('Predicted')
	ax.set_title(title)
	# plot x=y line
	ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'b--')
	# plot error bars
	# ax.errorbar(y_true, y_pred, yerr=y_pred - y_true, fmt="none", color="black")
	return fig


def plot_residuals(y_pred, y_true, title=None, **kwargs):
	"""Plot residuals."""
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.scatter(y_true, y_pred - y_true, marker='o', **kwargs)
	ax.set_xlabel('True')
	ax.set_ylabel('Residuals')
	ax.set_title(title)
	# plot x=0 line
	ax.plot([y_true.min(), y_true.max()], [0, 0], 'b--')
	# plot error bars
	# ax.errorbar(y_true, y_pred - y_true, yerr=y_pred - y_true, fmt="none", color="black")
	return fig
