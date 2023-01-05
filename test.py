'''#Imports
import cv2
import numpy as np
import urllib.request
import socket

s = socket.socket()         
 
s.bind(('0.0.0.0', 80 ))
s.listen(0)  

#Arduino Init
#Arduino = serial.Serial('COM3',9600)

#Load Camera and Weight
modelWeights = "Weights/weightA.onnx"
net = cv2.dnn.readNet(modelWeights)

#Constants
LED_CONSTANT = ""
LED_OFF = 'O'
LED_GREEN = 'W'
LED_YELLOW = 'I'
LED_RED = 'N'
PATH_NOPRESENCE = 'no_presence.png'
SENSE_DISTANCE = 121		#Distance in centimeters
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.51
POS_X = 440
POS_Y = 470
START_POINT = (430, 600)
END_POINT = (650, 450)

#Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

#Colors
WHITE = (255,255,255)
BLACK  = (0,0,0)
BLUE   = (255,178,50)
GREEN = (0,128,0)
YELLOW = (0,255,255)
RED = (0,0,255)

def led_off():
	"""Send an LED OFF command to Arduino"""
	off_command = LED_OFF.encode()

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    #Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]

    #Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)

    #Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, WHITE, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	#Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), (0,0,0), swapRB = True, crop=False)

	#Sets the input to the network.
	net.setInput(blob)

	#Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)

	return outputs


def post_process(input_image, outputs):
	#Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	#LED declaration
	led_indicator = LED_CONSTANT

	#Rows.
	rows = outputs[0].shape[1]

	image_height, image_width = input_image.shape[:2]

	#Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT

	#Iterate through 25200 detections.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]

		#Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			#Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)
				
	#Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		if(classes[class_ids[i]] == 'with_mask'):	#LED and bounding boxes color green
			led_indicator = LED_GREEN
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		elif(classes[class_ids[i]] == 'mask_incorrect'):	#LED and bounding boxes color yellow
			led_indicator = LED_YELLOW
			cv2.rectangle(input_image, (left, top), (left + width, top + height), YELLOW, 3*THICKNESS)
		elif(classes[class_ids[i]] == 'without_mask'):	#LED and bounding boxes color red
			led_indicator = LED_RED
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, 3*THICKNESS)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left + 100, top)
	return input_image, led_indicator

#Load class names.
classesFile = "class.names"
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().splitlines()

#Integrated Camera
while(True):
    img_resp = urllib.request.urlopen ('http://192.168.1.24/800x600.jpg')
    imgnp = np.array(bytearray(img_resp.read()), dtype = np.uint8)
    frame = cv2.imdecode(imgnp, -1)
    #ret, frame = vid.read()

    client, addr = s.accept()

    content = client.recv(32)

    count = str(content.decode().strip())
    print("People Count: " + count)
    #Process image.
    detections = pre_process(frame, net)
    img, led_info = post_process(frame.copy(), detections)

	#Image Show
    cv2.imshow('Live Feed', img)

	#Changes image scene
    #img = cv2.imread(PATH_NOPRESENCE)

    #Program Termination
    if cv2.waitKey(1) & 0xFF == ord('q'):
        led_off()
        break
  
# After the loop release the vid object
frame.release()
cv2.destroyAllWindows()'''
'''s={}
for i in range(1,17):
	s["s{0}_oc".format(i)] = False
	s["s{0}_uc".format(i)] = True

for i in range(1,17):
	print(i, s["s{0}_oc".format(i)])
	print(i, s["s{0}_uc".format(i)])'''

import socket

s = socket.socket()

s.bind(('0.0.0.0', 80))
s.listen(0)

while True:
	client, addr = s.accept()
	content = client.recv(64)
	count = str(content.decode().strip())
	print(count)
