import cv2, time
import numpy as np

net = cv2.dnn.readNet('./weights/yolov3.weights', './weights/yolov3.cfg')
classes = []

with open('./weights/obj.names', 'r') as f:
	classes = f.read().splitlines()

cap = cv2.VideoCapture('./uploads/fc5.mp4')
#frame = cv2.imread('./uploads/g3.jpg')


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


	cv2.imshow('Image', frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
	
cap.release
cv2.destroyAllWindows()
