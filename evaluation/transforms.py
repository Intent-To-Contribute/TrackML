import numpy as np
from sklearn.preprocessing import StandardScaler


def dbscan_trans(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	column1 = 'x' if replace else '1_db'
	column2 = 'y' if replace else '2_db'
	column3 = 'z' if replace else '3_db'

	r = np.sqrt(x**2 + y**2 + z**2)
	hits['column1'] = x/r
	hits['column2'] = y/r
	r = np.sqrt(x**2 + y**2)
	hits['column3'] = z/r

	hits[['column1', 'column2', 'column3']] = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	X = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)

	return X

def spherical(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	column1 = 'x' if replace else '1_sph'
	column2 = 'y' if replace else '2_sph'
	column3 = 'z' if replace else '3_sph'

	r = np.sqrt(x**2 + y**2 + z**2)
	xy_r = np.sqrt(x**2 + y**2)
	hits['column1'] = r
	hits['column2'] = np.arctan2(xy_r, z)
	hits['column3'] = np.arctan2(y, x)

	hits[['column1', 'column2', 'column3']] = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	X = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	return X

def cylindrical(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	column1 = 'x' if replace else '1_cyl'
	column2 = 'y' if replace else '2_cyl'
	column3 = 'z' if replace else '3_cyl'

	xy_r = np.sqrt(x**2 + y**2)
	hits['column1'] = xy_r
	hits['column2'] = np.arcsin(y/xy_r)*(x >= 0)*np.invert((x == 0)*(y == 0)) + \
					(-np.arcsin(y/xy_r) + np.pi)*(x < 0)
	hits['column3'] = z

	hits[['column1', 'column2', 'column3']] = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	X = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	return X


def normalize(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	r = np.sqrt(x**2 + y**2 + z**2)
	hits['1_norm'] = x/r
	hits['2_norm'] = y/r
	hits['3_norm'] = z/r
	hits['4_norm'] = r

	hits[['1_norm', '2_norm', '3_norm', '4_norm']] = ss.fit_transform(hits[['1_norm', '2_norm', '3_norm', '4_norm']].values)
	X = ss.fit_transform(hits[['1_norm', '2_norm', '3_norm', '4_norm']].values)
	return X


def standard(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	column1 = 'x' if replace else '1_ss'
	column2 = 'y' if replace else '2_ss'
	column3 = 'z' if replace else '3_ss'

	hits['column1'] = x
	hits['column2'] = y
	hits['column3'] = z

	hits[['column1', 'column2', 'column3']] = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	X = ss.fit_transform(hits[['column1', 'column2', 'column3']].values)
	return X

def identity(hits, replace=False):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values

	column1 = 'x' if replace else '1_id'
	column2 = 'y' if replace else '2_id'
	column3 = 'z' if replace else '3_id'

	hits['column1'] = x
	hits['column2'] = y
	hits['column3'] = z

