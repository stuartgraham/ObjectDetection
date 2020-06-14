# Object Detection from MQTT messages using OpenCV
OpenCV `dnn` module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow. 

Tested to work with YOLOv3 and v4 predefined models
 
### Environment variables
Pass the following environment vairables to execution environment
| Settings | Description | Inputs |
| :----: | --- | --- |
| `CONFIG` | dnn model config file | `yolov3.cfg` |
| `WEIGHTS` | dnn model weights file | `yolov3.weights` |
| `CLASSES` | dnn model classes file | `coco_classes.txt` |
| `MODEL_WEIGHTS_URL` | URL for dnn model config file | `https://pjreddie.com/media/files/yolov3.weights` |
| `MODEL_CFG_URL` | URL for dnn model weights file | `https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg` |
| `INPUT_PATH` | Sub directory with input files | `input` |
| `OUTPUT_PATH` | Sub directory with output files | `output` |
| `MQTT_BROKER` | MQTT Broker address | `mqtt.test.local` |
| `MQTT_PORT` | MQTT Broker port | `1883` |
| `MQTT_SUB_TOPIC` | MQTT Topic to subscribe to | `test/messages` |
| `MQTT_PUB_TOPIC` | MQTT Topic to publish to | `test/messages` |

### Requirements
```sh
pip install -p requirements.txt
```

### Execution 
```sh
python3 .\main.py
```

### Docker Compose
```sh 
objectdetection:
    image: stuartgraham/objectdetection
    container_name: objectdetection
    environment:
        - CONFIG=yolov3.cfg
        - WEIGHTS=yolov3.weights
        - CLASSES=classes.txt
        - INPUT_PATH=input
        - OUTPUT_PATH=output
        - MQTT_BROKER=mqtt.test.local
        - MQTT_PORT=1883
        - MQTT_PUB_TOPIC=cctv/output
        - MQTT_SUB_TOPIC=cctv/input
        - MODEL_WEIGHTS_URL=https://pjreddie.com/media/files/yolov3.weights
        - MODEL_CFG_URL=https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    volumes:
        - objectdetection-models:/ObjectDetection/models
        - objectdetection-storage:/ObjectDetection/output
        - cctv-input-jpg:/ObjectDetection/input:ro
    restart: always
```
