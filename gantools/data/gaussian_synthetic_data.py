import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix as RAND_COV

def generate_cube(cube_dim):
	'''3D gaussians : x,y,z independent variables, f dependent variable
	'''

	# generate the x, y, z values
	lim = 1
	x, y, z = np.mgrid[0:lim:complex(0, cube_dim), 0:lim:complex(0, cube_dim), 0:lim:complex(0, cube_dim)] # z changes fastest, then y, then x
	# Need an (N, 3) array of (x, y, z) triplets.
	xyz = np.column_stack([x.flat, y.flat, z.flat])

	f = 0
	num_gaussians = 20;
	for _ in range(num_gaussians):
		mu = np.random.rand(3)
		C = RAND_COV(3)
		#C = np.diag([1, 1, 1])
		v = xyz - mu
		sigma = 0.1
		f += np.exp(-np.sum(v * (C @ v.T).T, axis=1)/sigma**2)
		#f += multivariate_normal.pdf(xyz, mean=mu, cov=C)

	f = f.reshape(x.shape)  # THE CUBE OF DATA WITH DIM = [cube_dim, cube_dim, cube_dim] [x_index, y_index, z_index]
	return f

def generate_cubes(nsamples, cube_dim):
	cubes = []
	for i in range(nsamples):
		cubes.append(generate_cube(cube_dim))

	cubes = np.array(cubes)
	cubes.resize([nsamples, cube_dim, cube_dim, cube_dim])
	return cubes


def generate_square(square_dim):
	'''2D gaussians : x,y independent variables, f dependent variable
	'''

	# generate the x, y values
	lim = 1
	x, y = np.mgrid[0:lim:complex(0, square_dim), 0:lim:complex(0, square_dim)] # y changes fastest, then x
	# Need an (N, 2) array of (x, y)
	xy = np.column_stack([x.flat, y.flat])

	f = 0
	num_gaussians = 5;
	for _ in range(num_gaussians):
		mu = np.random.rand(2)
		C = RAND_COV(2)
		v = xy - mu
		sigma = 0.25
		f += np.exp(-np.sum(v * (C @ v.T).T, axis=1)/sigma**2)
		#f += multivariate_normal.pdf(xy, mean=mu, cov=C)


	f = f.reshape(x.shape)  # THE CUBE OF DATA WITH DIM = [square_dim, square_dim]
	return f

def generate_squares(nsamples, square_dim):
	squares = []
	for i in range(nsamples):
		squares.append(generate_square(square_dim))

	squares = np.array(squares)
	squares.resize([nsamples, square_dim, square_dim])
	return squares