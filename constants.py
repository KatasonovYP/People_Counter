from initialization import *

args = parser()

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
CLASS_NAMES = load_class_names(args['names'])
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

