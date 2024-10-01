import xml.etree.ElementTree as ET

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