import cv2
import argparse
import numpy as np
import os
import paho.mqtt.client as paho
import requests
import json
import logging

# # .ENV FILE FOR TESTING
#if os.path.exists('.env'):
#    from dotenv import load_dotenv
#    load_dotenv()

# GLOBALS
CONFIG = os.environ.get('CONFIG','')
WEIGHTS = os.environ.get('WEIGHTS','')
CLASSES = os.environ.get('CLASSES','')
INPUT_PATH = os.environ.get('INPUT_PATH','output')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH','samples')
MQTT_BROKER = os.environ.get('MQTT_BROKER','')
MQTT_PORT = int(os.environ.get('MQTT_PATH', 1883))
MQTT_PUB_TOPIC = os.environ.get('MQTT_PUB_TOPIC','')
MQTT_SUB_TOPIC = os.environ.get('MQTT_SUB_TOPIC','')
MODEL_WEIGHTS_URL = os.environ.get('MODEL_WEIGHTS_URL','')
MODEL_CFG_URL = os.environ.get('MODEL_CFG_URL','')
LOGGING = os.environ.get('LOGGING','INFO')

#TRANFORM GLOBALS
CONFIG = 'models/' + CONFIG
WEIGHTS = 'models/' + WEIGHTS
CLASSES = 'models/' + CLASSES
CONFIG = os.path.join(os.getcwd(), CONFIG)
WEIGHTS = os.path.join(os.getcwd(), WEIGHTS)
CLASSES = os.path.join(os.getcwd(),  CLASSES)
BASE_DIR = os.getcwd()
INPUT_PATH = os.path.join(BASE_DIR, INPUT_PATH)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_PATH)

def check_model_files():
    if not os.path.exists(CONFIG):
        logging.info('CONFIG File not found, downloading')
        url = MODEL_CFG_URL
        r = requests.get(url, allow_redirects=True)
        open(CONFIG, 'wb').write(r.content)

    if not os.path.exists(WEIGHTS):
        logging.info('WEIGHTS File not found, downloading')
        url = MODEL_WEIGHTS_URL
        r = requests.get(url, allow_redirects=True)
        open(WEIGHTS, 'wb').write(r.content)

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return label

def examime_image(image):
    global classes
    global COLORS
    os.chdir(INPUT_PATH)
    image_name = image
    image = cv2.imread(image)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None

    with open(CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    metadata = {}
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        category = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        metadata = {'image': image_name, 'category': category, 'confidence': float("%0.3f" % (confidences[i]))}
        logging.info(metadata)

    if metadata:
        push_mqtt_message(metadata)
        os.chdir(OUTPUT_PATH)
        cv2.imwrite(image_name, image)
        logging.info('Image processed')

# SUB MQTT
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))
    client.subscribe(MQTT_SUB_TOPIC)

def on_message(client, userdata, msg):
    logging.info("Recieved message {} convert to json".format(str(msg.payload))) 
    message = json.loads(msg.payload)
    if message['pathToImage']:
        image_path = message['pathToImage']
        logging.info('Received valid message, processing {}'.format(image_path))
        examime_image(image_path)
    else:
        logging.info('Received in valid message, processing')

# PUB MQTT
def on_publish(client,userdata,result,rc):  
    logging.info("Connected with result code "+str(rc))

def push_mqtt_message(message):
    client1 = paho.Client("object-detector")                        
    client1.on_publish = on_publish                         
    client1.connect(MQTT_BROKER, MQTT_PORT)                                
    client1.publish(MQTT_PUB_TOPIC, json.dumps(message))           

def main():
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("STARTING Object Detection")
    check_model_files()
    client = paho.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    run = True
    while run:
        client.loop()
    
# Main Exectution
main()
