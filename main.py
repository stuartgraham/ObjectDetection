import cv2
import argparse
import numpy as np
import os
import paho.mqtt.client as paho
import paho.mqtt.publish as publish
import requests
import json
import logging
import time

# .ENV FILE FOR TESTING
#if os.path.exists('.env'):
#    from dotenv import load_dotenv
#    load_dotenv()

# GLOBALS
TESTING = False
CONFIG = os.environ.get('CONFIG','')
WEIGHTS = os.environ.get('WEIGHTS','')
CLASSES = os.environ.get('CLASSES','')
INPUT_PATH = os.environ.get('INPUT_PATH','input')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH','output')
MQTT_BROKER = os.environ.get('MQTT_BROKER','')
MQTT_PORT = int(os.environ.get('MQTT_PATH', 1883))
MQTT_PUB_TOPIC = os.environ.get('MQTT_PUB_TOPIC','')
MQTT_SUB_TOPIC = os.environ.get('MQTT_SUB_TOPIC','')
MODEL_WEIGHTS_URL = os.environ.get('MODEL_WEIGHTS_URL','')
MODEL_CFG_URL = os.environ.get('MODEL_CFG_URL','')

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
    cv2.putText(img, label.upper(), (x-10,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
    return label

def examine_image(image):
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

    metadatas = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        category = draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        metadatas.append({'image': image_name, 'category': category, 'confidence': float("%0.3f" % (confidences[i]))})

    if metadatas:
        #print(metadatas)
        max_confidence = metadatas[0]
        for i in metadatas:
            if i['confidence'] > max_confidence['confidence']:
                max_confidence = i
        
        metadata = max_confidence
        os.chdir(OUTPUT_PATH)
        cv2.imwrite(image_name, image)
        #print(metadata)
        push_mqtt_message(metadata)
        logging.info('Image processed : {}'.format(metadata))
        #print("#"*30)

# SUB MQTT
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))
    client.subscribe(MQTT_SUB_TOPIC)

def on_message(client, userdata, msg):
    logging.debug("Recieved message {} convert to json".format(str(msg.payload))) 
    message = json.loads(msg.payload)
    if message['pathToImage']:
        image_path = message['pathToImage']
        logging.info('Received valid message, processing {}'.format(image_path))
        examine_image(image_path)
    else:
        logging.info('Received in valid message, processing')

# PUB MQTT
def push_mqtt_message(message):
    publish.single(MQTT_PUB_TOPIC,
        payload=json.dumps(message),
        hostname=MQTT_BROKER,
        client_id="object-detector-pub",
        port=MQTT_PORT)

def testing():
    start_time = time.time()
    file_names = os.listdir(INPUT_PATH)
    os.chdir(INPUT_PATH)
    for file_name in file_names:
        examine_image(file_name)
    print("--- %s seconds ---" % (time.time() - start_time))

def main():
    logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("STARTING Object Detection")
    check_model_files()
    sub_client = paho.Client("object-detector-sub")
    sub_client.on_connect = on_connect
    sub_client.on_message = on_message
    sub_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    sub_client.loop_forever()

    
# Main Exectution
if TESTING == True:
    testing()
else:
    main()
