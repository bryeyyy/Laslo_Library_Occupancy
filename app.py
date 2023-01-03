from flask import Flask,render_template,Response, jsonify, request, redirect, url_for
import time
import cv2
import threading
from turbo_flask import Turbo
import numpy as np
import urllib.request
import socket
import cv2
import numpy as np
import urllib.request
import socket
import time
import threading

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

occupy = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

print(cv2.__version__)
s = socket.socket()         
 
s.bind(('0.0.0.0', 80 ))
s.listen(0)

#Load Camera and Weight
modelWeights = "Weights/occ.onnx"
net = cv2.dnn.readNet(modelWeights)

#Load class names.
classesFile = "class.names"
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().splitlines()

#to include css files and other misc files, create that folder in the
#directory, then include it in the flask init
#example : app = Flask(__name__,template_folder='./frontend/templates',static_folder='./frontend/static')

#init server instance, SERVER_NAME = your url:port , needed for url_for referencing
app = Flask(__name__,template_folder='templates',static_folder='static')

#change server name depending on your LOCAL IP Address -> cmd -> ipconfig
app.config['SERVER_NAME'] = '192.168.1.2:5000'  
#app.config['SECRET_KEY'] = 'password1234567890'


#init turbo instance for websocket update stuff
turbo = Turbo(app)


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

	#Occupancy Variables
	s1_oc = False
	s1_uc = False
	s10_oc = False
	s10_uc = False
	s11_oc = False
	s11_uc = False
	s12_oc = False
	s12_uc = False
	s13_oc = False
	s13_uc = False
	s14_oc = False
	s14_uc = False
	s15_oc = False
	s15_uc = False
	s16_oc = False
	s16_uc = False
	s2_oc = False
	s2_uc = False
	s3_oc = False
	s3_uc = False
	s4_oc = False
	s4_uc = False
	s5_oc = False
	s5_uc = False
	s6_oc = False
	s6_uc = False
	s7_oc = False
	s7_uc = False
	s8_oc = False
	s8_uc = False
	s9_oc = False
	s9_uc = False
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
		if(classes[class_ids[i]] == 's1_oc'):
			s1_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's1_uc'):
			s1_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's2_oc'):
			s2_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's2_uc'):
			s2_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's3_oc'):
			s3_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's3_uc'):
			s3_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top +	height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's4_oc'):
			s4_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's4_uc'):
			s4_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's5_oc'):
			s5_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's5_uc'):
			s5_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's6_oc'):
			s6_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's6_uc'):
			s6_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's7_oc'):
			s7_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's7_uc'):
			s7_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's8_oc'):
			s8_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's8_uc'):
			s8_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's9_oc'):
			s9_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's9_uc'):
			s9_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's10_oc'):
			s10_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's10_uc'):
			s10_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's11_oc'):
			s11_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's11_uc'):
			s11_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's12_oc'):
			s12_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's12_uc'):
			s12_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's13_oc'):
			s13_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's13_uc'):
			s13_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's14_oc'):
			s14_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's14_uc'):
			s14_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's15_oc'):
			s15_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's15_uc'):
			s15_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		if(classes[class_ids[i]] == 's16_oc'):
			s16_oc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, -1)
		if(classes[class_ids[i]] == 's16_uc'):
			s16_uc = True
			cv2.rectangle(input_image, (left, top), (left + width, top + height), GREEN, 3*THICKNESS)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left + 100, top)
	return input_image,	s1_oc, s1_uc, s10_oc,s10_uc,s11_oc,s11_uc,s12_oc,s12_uc,s13_oc,s13_uc,s14_oc,s14_uc,s15_oc,s15_uc,s16_oc,s16_uc,s2_oc,s2_uc,s3_oc,s3_uc,s4_oc,s4_uc,s5_oc,s5_uc,s6_oc,s6_uc,s7_oc,s7_uc,s8_oc,s8_uc,s9_oc,s9_uc

#generate video frames before sending to the front-end
def gen_frames():
    while True:

        img_resp = urllib.request.urlopen ('http://192.168.1.137/800x600.jpg')
        imgnp = np.array(bytearray(img_resp.read()), dtype = np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        detections = pre_process(frame, net)
        img, s1_oc, s1_uc, s10_oc,s10_uc,s11_oc,s11_uc,s12_oc,s12_uc,s13_oc,s13_uc,s14_oc,s14_uc,s15_oc,s15_uc,s16_oc,s16_uc,s2_oc,s2_uc,s3_oc,s3_uc,s4_oc,s4_uc,s5_oc,s5_uc,s6_oc,s6_uc,s7_oc,s7_uc,s8_oc,s8_uc,s9_oc,s9_uc = post_process(frame.copy(), detections)

        if(s1_oc == True and s1_uc == False):
            occupy[0]=True
        if(s1_oc == False and s1_uc == True):
            occupy[0] = False
        if(s2_oc == True and s2_uc == False):
            occupy[1] = True
        if(s2_oc == False and s2_uc == True):
            occupy[1] = False
        if(s3_oc == True and s3_uc == False):
            occupy[2] = True
        if(s3_oc == False and s3_uc == True):
            occupy[2] = False
        if(s4_oc == True and s4_uc == False):
            occupy[3] = True
        if(s4_oc == False and s4_uc == True):
            occupy[3] = False
        if(s5_oc == True and s5_uc == False):
            occupy[4] = True
        if(s5_oc == False and s5_uc == True):
            occupy[4] = False
        if(s6_oc == True and s6_uc == False):
            occupy[5] = True
        if(s6_oc == False and s6_uc == True):
            occupy[5] = False
        if(s7_oc == True and s7_uc == False):
            occupy[6] = True
        if(s7_oc == False and s7_uc == True):
            occupy[6] = False
        if(s8_oc == True and s8_uc == False):
            occupy[7] = True
        if(s8_oc == False and s8_uc == True):
            occupy[7] = False
        if(s9_oc == True and s9_uc == False):
            occupy[8] = True
        if(s9_oc == False and s9_uc == True):
            occupy[8] = False
        if(s10_oc == True and s10_uc == False):
            occupy[9] = True
        if(s10_oc == False and s10_uc == True):
            occupy[9] = False
        if(s11_oc == True and s11_uc == False):
            occupy[10] = True
        if(s11_oc == False and s11_uc == True):
            occupy[10] = False
        if(s12_oc == True and s12_uc == False):
            occupy[11] = True
        if(s12_oc == False and s12_uc == True):
            occupy[11] = False
        if(s13_oc == True and s13_uc == False):
            occupy[12] = True
        if(s13_oc == False and s13_uc == True):
            occupy[12] = False
        if(s14_oc == True and s14_uc == False):
            occupy[13] = True
        if(s14_oc == False and s14_uc == True):
            occupy[13] = False
        if(s15_oc == True and s15_uc == False):
            occupy[14] = True
        if(s15_oc == False and s15_uc == True):
            occupy[14] = False
        if(s16_oc == True and s16_uc == False):
            occupy[15] = True
        if(s16_oc == False and s16_uc == True):
            occupy[15] = False

        ret,buffer=cv2.imencode('.jpg',img)
        img=buffer.tobytes()

        yield(b'--img\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


# ROUTING TO WEBSITE INTERFACE

#index - main route for rendering the home page
@app.route('/')
def index():
    return render_template('test.html')

@app.route('/seatdata', methods=['GET', 'POST'])
def seatdata():
    if request.method == 'GET':
        datasend = occupy
    return jsonify(datasend)

#route for sending video data to html doc
@app.route('/video')
def video():
    return Response(gen_frames(),mimetype = 'multipart/x-mixed-replace; boundary=img')

@app.route('/loginadminax', methods=['GET', 'POST'])
def loginadminax():
		error = None
		if request.method == 'POST':
			if request.form['username'] != 'admin' or request.form['password'] != 'admin':
				error = 'Invalid Credentials. Please try again.'
			else:
				return redirect(url_for('adminax'))
		return render_template('login.html', error=error)

@app.route('/adminax')
def adminax():
    return render_template('adminaccess.html')


@app.route('/postresponsechat', methods=['GET', 'POST'])
def postresponsechat():
		usermessage  = request.form['usermessage']
		print(usermessage)
		return str(usermessage)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    return render_template('chatbot.html')


#ref1: https://blog.miguelgrinberg.com/post/dynamically-update-your-flask-web-pages-using-turbo-flask
#ref2: https://turbo-flask.readthedocs.io/en/latest/quickstart.html#how-to-use

#to update the server periodically, use app.app_context, then turbo methods for updating html innerText
def update_seats():
    with app.app_context():
        while True:
            client, addr = s.accept()
            content = client.recv(32)
            count = str(content.decode().strip())

            time.sleep(2)

            load3 = count #for people count
            turbo.push(turbo.update(load3, 'libcount'))


# refer to ref1 for details, forces the flask instance to load the func update_seats at the start of the server runtime
@app.before_first_request
def before_first_request():
    threading.Thread(target=update_seats).start()

# to run the server
if __name__=="__main__":
    app.run()