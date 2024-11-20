import xml.etree.ElementTree as ET
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd

def mps2kph(mps : float) -> float:
    return 3.6 * mps

def kph2mps(kph : float) -> float:
    return kph/3.6

def parse_net(file):
	data = {}
	tree = ET.parse(file)
	root = tree.getroot()

	for e in [x for x in root if x.tag == 'edge' and not ':' in x.attrib['id'] and 's' in x.attrib['id']]:
		lanes = {}
		for ls in e:
			lanes[ls.attrib['id']] = float(ls.attrib['length'])
		data[e.attrib['id']] = lanes
	return data

def passenger_polygon(
		deg : float, 
		center : Tuple[float, float] = (0,0)
	) -> Polygon:
	"""
	Creates a polygon for the boundary of a SUMO passenger car.

	:: PARAMETERS ::
	deg : vehicle rotation (in degrees)
	center : Center of the SUMO vehicle

	:: RETURN ::
	A Shapely Polygon which marks the boundary of a sumo passenger vehicle.
	"""
	rad = np.deg2rad(deg)
	points = np.array([
		(0.5, -0.31063829787234043),
		(0.42040185471406494, -0.44468085106382976),
		(0.23647604327666152, -0.5),
		(-0.45440494590417313, -0.5),
		(-0.5, -0.4085106382978723),
		(-0.5, 0.4085106382978723),
		(-0.45440494590417313, 0.5),
		(0.23647604327666152, 0.5),
		(0.42040185471406494, 0.44468085106382976),
		(0.5, 0.31063829787234043)
	])
	scale_width = 1.8 # m
	scale_length = 5  # m

	# Create the polygon
	points = np.array([
		points.T[0] * scale_length - 2.5,
		points.T[1] * scale_width
	]).T
	points = points + center

	# Create the 2D rotation matrix
	rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)],
								[np.sin(rad), np.cos(rad)]])
	
	# Step 1: Translate points to the origin
	translated_points = points - center
	
	# Step 2: Rotate the points
	rotated_points = np.dot(translated_points, rotation_matrix.T)  # .T to transpose the matrix
	
	# Step 3: Translate points back to the original center
	final_points = rotated_points + center
	return Polygon(final_points)

def plot_polygon(polygon : Polygon):
	# Extract the x and y coordinates of the Polygon
	x, y = polygon.exterior.xy

	# Plot the Polygon using matplotlib
	plt.figure()
	plt.plot(x, y, color='blue', linewidth=2, linestyle='-', marker='o')
	plt.fill(x, y, color='skyblue', alpha=0.5)  # Optionally fill the Polygon
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.grid()
	plt.gca().set_aspect('equal')
	plt.show()
	return 

def plot_car_polygons(dut_polygon : Polygon, foe_polygon : Polygon):
	# plt.clf()
	plt.figure()

	plt.plot(*dut_polygon.exterior.xy, 
		  color='blue', linewidth=2, linestyle='-', marker='o')
	plt.fill(*dut_polygon.exterior.xy, color='skyblue', alpha=0.5) 
	
	plt.plot(*foe_polygon.exterior.xy, 
		  color='red', linewidth=2, linestyle='-', marker='o')
	plt.fill(*foe_polygon.exterior.xy, color='lightcoral', alpha=0.5) 

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.grid()
	plt.gca().set_aspect('equal')
	plt.show()
	return

def describe_as_latex(df : pd.DataFrame) -> str:
	"""
	Summarizes a DataFrame @df in latex format.
	"""
	stats = ["count", "mean", "std", "min", "max"]
	df = df.copy()
	df = df.describe().T[["count", "mean", "std", "min", "max"]]
	df = df.round(decimals=3)

	print(df)
	print()

	alignment = "rlrr@{.}lr@{.}lr@{.}lr@{.}l"
	msg = "\\begin{tabular}{%s}\\toprule\n" % alignment
	msg += "\t & Feature "
	
	for feat in stats:
		if feat == "count":
			msg += "& Count "
		else:
			msg += "& \\multicolumn{2}{c}{%s} " % feat.capitalize()
		continue
	msg += "\\\\\\midrule\n"

	for i in range(len(df.index)):
		s = df.iloc[i]
		msg += " &"
		msg += " %s " % s.name

		for feat in stats:
			left , right = str(s[feat]).split(".")
			left = "{:,}".format(int(left))
			if feat == "count":
				msg += "& %s " % left
			else: 
				msg += "& %s&%s " % (left,right)
			continue

		msg += "\\\\\n"
		continue
		
	msg += "\\bottomrule\n"
	msg += "\\end{tabular}\n"

	print(msg)



	return