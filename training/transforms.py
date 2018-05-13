import numpy as np
from sklearn.preprocessing import StandardScaler


def dbscan_trans(hits):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values

	ss = StandardScaler()
	r = np.sqrt(x**2 + y**2 + z**2)
	hits['1_db'] = x/r
	hits['2_db'] = y/r

	r = np.sqrt(x**2 + y**2)
	hits['3_db'] = z/r

	hits[['1_db', '2_db', '3_db']] = ss.fit_transform(hits[['1_db', '2_db', '3_db']].values)
	X = ss.fit_transform(hits[['1_db', '2_db', '3_db']].values)
	return X

def spherical(hits):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	r = np.sqrt(x**2 + y**2 + z**2)
	xy_r = np.sqrt(x**2 + y**2)
	hits['1_sph'] = r
	hits['2_sph'] = np.arctan2(xy_r, z)
	hits['3_sph'] = np.arctan2(y, x)

	hits[['1_sph', '2_sph', '3_sph']] = ss.fit_transform(hits[['1_sph', '2_sph', '3_sph']].values)
	X = ss.fit_transform(hits[['1_sph', '2_sph', '3_sph']].values)
	return X

def cylindrical(hits):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	xy_r = np.sqrt(x**2 + y**2)
	hits['1_cyl'] = xy_r
	hits['2_cyl'] = np.arcsin(y/xy_r)*(x >= 0)*np.invert((x == 0)*(y == 0)) + \
					(-np.arcsin(y/xy_r) + np.pi)*(x < 0)
	hits['3_cyl'] = z

	hits[['1_cyl', '2_cyl', '3_cyl']] = ss.fit_transform(hits[['1_cyl', '2_cyl', '3_cyl']].values)
	X = ss.fit_transform(hits[['1_cyl', '2_cyl', '3_cyl']].values)
	return X


def normalize(hits):
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


def standard(hits):
	x = hits.x.values
	y = hits.y.values
	z = hits.z.values
	ss = StandardScaler()

	hits['1_ss'] = x
	hits['2_ss'] = y
	hits['3_ss'] = z

	hits[['1_ss', '2_ss', '3_ss']] = ss.fit_transform(hits[['1_ss', '2_ss', '3_ss']].values)
	X = ss.fit_transform(hits[['1_ss', '2_ss', '3_ss']].values)
	return X