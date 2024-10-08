import xml.etree.ElementTree as ET
import math
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon

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