import numpy as np
import math
import pandas as pd

def add_clutter(df):
	cell_x = df['Cell X'][0]
	cell_y = df['Cell Y'][0]
	columns = list(range(1, 21))
	df_count = pd.DataFrame(0, index=list(df.index), columns=columns)
	n = [2,3,4,1.7,5,2.5]
	all_n = []
	for i in df.index:
		x = df.loc[i]['X']
		y = df.loc[i]['Y']
		count = df[(df['X'] > min(cell_x, x)) & (df['X'] < max(cell_x, x)) &
				   (df['Y'] > min(cell_y, y)) &
				   (df['Y'] < max(cell_y, y))]['Clutter Index'].value_counts()
		df_count.loc[i] = count
		df_count.fillna(0, inplace=True)
		one_count = df_count.loc[i]
		one_n = [0, 0, 0, 0, 0, 0]
		one_n[0] = sum(df_count[[1, 2, 3, 4, 5, 6, 7, 8, 9]].loc[i])
		one_n[1] = sum(df_count[[15, 18, 20]].loc[i])
		one_n[2] = sum(df_count[[13, 16]].loc[i])
		one_n[3] = sum(df_count[[17, 19]].loc[i])
		one_n[4] = sum(df_count[[10, 11]].loc[i])
		one_n[5] = sum(df_count[[14, 12]].loc[i])
		all_n.append(n[one_n.index(max(one_n))])
	df_count['n'] = all_n
	return df_count

def calculate_attr(df):
	df['eff_height'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
	df['airline_distance'] = np.sqrt((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)
	theta = df['Electrical Downtilt'][0] + df['Mechanical Downtilt'][0]
	df['delta_eff_height'] = df['eff_height'] - (df['airline_distance'] * math.tan(theta))
	df['emit_build_distance'] = np.sqrt((df['eff_height'] -df['Building Height'])**2 +df['airline_distance']**2)
	df['power_dis_rate'] = df['RS Power'] / df['airline_distance']
	df['power_area_rate'] = df['RS Power'] / (math.pi * df['airline_distance'] * df['airline_distance'])
	df['delta_reflect'] = df['eff_height'] * 1.62 / df['airline_distance']
	df['path_cost2'] = df['n'] * np.log (df['emit_build_distance'] / df['airline_distance'])
	angle = df['Azimuth'][0]
	x0 = df['Cell X'][0]
	y0 = df['Cell Y'][0]
	if_covered = []
	if ((angle == 0) | (angle == 360)):
		for i in range(len(df)):
			if (df['X'][i] >= x0):
				if_covered.append(1)
			else:
				if_covered.append(0)
	elif ((angle > 0 & angle < 90) | (angle > 90 & angle < 180)):
		slope = 1 / math.tan(2 * angle / 360 * math.pi)
		for i in range(len(df)):
			tmp_y = slope * (df['X'][i] - x0) + y0
			if (df['Y'][i] <= tmp_y):
				if_covered.append(1)
			else:
				if_covered.append(0)
	elif (angle == 90):
		for i in range(len(df)):
			if (df['y'][i] <= y0):
				if_covered.append(1)
			else:
				if_covered.append(0)
	elif ((angle > 180 & angle < 270) | (angle > 270 & angle < 360)):
		slope = 1 / math.tan(2 * angle / 360 * math.pi)
		for i in range(len(df)):
			tmp_y = slope * (df['X'][i] - x0) + y0
			if (df['Y'][i] >= tmp_y):
				if_covered.append(1)
			else:
				if_covered.append(0)
	elif (angle == 180):
		for i in range(len(df)):
			if (df['X'][i] <= x0):
				if_covered.append(1)
			else:
				if_covered.append(0)
	elif (angle == 270):
		for i in range(len(df)):
			if (df['y'][i] >= y0):
				if_covered.append(1)
			else:
				if_covered.append(0)
	else:
		print("Angle EXCEPTION!")
		if_covered.append(0)
	df['if_covered'] = if_covered
	
	new_row_sheet = df[['eff_height','airline_distance','delta_eff_height',
						'emit_build_distance','power_dis_rate',
						'power_area_rate','delta_reflect','path_cost2','if_covered']]
	return new_row_sheet