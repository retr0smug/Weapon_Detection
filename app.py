# import the necessary packages
import numpy as np
import time, cv2, os, random, io
from flask import Flask, request, Response,render_template
#import binascii
import io as StringIO
from io import BytesIO
from PIL import Image
from cv2 import imencode
from flask_bootstrap import Bootstrap
from flask_ngrok import run_with_ngrok


route=os.path.dirname(os.path.abspath(__file__))



# construct the argument parse and parse the arguments

labelsPath="weights/obj.names"
configPath="weights/yolov3.cfg"
weightsPath="weights/yolov3.weights"



# Initialize the Flask app
app = Flask(__name__)
run_with_ngrok(app)  



#####-----------------------------------IMAGE-----------------------------------#####

@app.route('/')
def home():
    return render_template('index.html')

# route http posts to this method
@app.route('/img_output', methods=['POST'])
def img_output():
    img = request.files["file"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   
    #dam=0
	# load the COCO class labels our YOLO model was trained on
    LABELS = open(labelsPath).read().strip().split("\n")
	# initialize a list of colors to represent each possible class label
    np.random.seed(11)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")
	# derive the paths to the YOLO weights and model configuration
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	# load our input image and grab its spatial dimensions
	# ************************************\
    #image = cv2.imread("static/test.jpeg")
	#************************************
    (H, W) = image.shape[:2]
	# determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
	# show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
	# loop over each of the layer outputs
    for output in layerOutputs:
		# loop over each of the detections
	    for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
		    if confidence > 0.5 :
				# scale the bounding box coordinates back relative to the
				# size of the image, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
			    boxes.append([x, y, int(width), int(height)])
			    confidences.append(float(confidence))
			    classIDs.append(classID)
	# apply non-maxima suppression to sup.read()press weak, overlapping bounding
	# boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)
	# ensure at least one detection exists
    if len(idxs) > 0:
		# loop over the indexes we are keeping
	    for i in idxs.flatten():
			# extract the bounding box coordinates
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
		    color = [int(c) for c in COLORS[classIDs[i]]]
		    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)

    a=random.randrange(1, 5000, 1)
    b= str(a)+".png"

    dest=os.path.join(route,"output",b)
    cv2.imwrite(dest,image)
    imgval="../output/{}".format(b)
    cv2.imshow("Image", image)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        return render_template('index.html')

#####-----------------------------------VIDEO-----------------------------------#####
##########################################
##########################################

@app.route('/vidproc', methods=['GET', 'POST'])
def vidproc():
    return render_template('video.html')


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():

    net = cv2.dnn.readNet('./weights/yolov3.weights', './weights/yolov3.cfg')
    classes = []

    with open('./weights/obj.names', 'r') as f:
        classes = f.read().splitlines()

    vid_input = request.files["file"]

    print("############File loaded############")
    cap = cv2.VideoCapture(vid_input)


    while True:
        _, frame = cap.read()

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2]* width)
                    h = int(detection[3]* height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x,y,w,h])
                    confidences.append( (float(confidence)) )
                    class_ids.append(class_id)

        print(len(boxes))
        indexes = cv2.dnn.NMSBoxes( boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=( len(boxes), 3 ) )

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str( classes[class_ids[i]] )
            confidence = str( round(confidences[i], 2) )
            color = colors[i]
            cv2.rectangle( frame, (x, y), (x+w, y+h), color, 2 )
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2 )


        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cap.release
    cv2.destroyAllWindows()
    return render_template('video.html')
    
#####-----------------------------------WEBCAM-----------------------------------#####
@app.route('/realproc', methods=['GET', 'POST'])
def realproc():
    return render_template('realtime.html')

##imports
from webcam import *
from camera_settings import *
Bootstrap(app)

check_settings()
VIDEO = VideoStreaming()

@app.route('/webcam_feed')
def webcam_feed():
    return Response(VIDEO.show(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Button requests called from ajax
@app.route('/request_preview_switch')
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print('*'*10, VIDEO.preview)
    return "nothing"

@app.route('/request_flipH_switch')
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    print('*'*10, VIDEO.flipH)
    return "nothing"

@app.route('/request_model_switch')
def request_model_switch():
    VIDEO.detect = not VIDEO.detect
    print('*'*10, VIDEO.detect)
    return "nothing"

@app.route('/request_exposure_down')
def request_exposure_down():
    VIDEO.exposure -= 1
    print('*'*10, VIDEO.exposure)
    return "nothing"

@app.route('/request_exposure_up')
def request_exposure_up():
    VIDEO.exposure += 1
    print('*'*10, VIDEO.exposure)
    return "nothing"

@app.route('/request_contrast_down')
def request_contrast_down():
    VIDEO.contrast -= 4
    print('*'*10, VIDEO.contrast)
    return "nothing"

@app.route('/request_contrast_up')
def request_contrast_up():
    VIDEO.contrast += 4
    print('*'*10, VIDEO.contrast)
    return "nothing"

@app.route('/reset_camera')
def reset_camera():
    STATUS = reset_settings()
    print('*'*10, STATUS)
    return "nothing"
  



    # start flask app
if __name__ == '__main__':
    app.run(debug=True)
