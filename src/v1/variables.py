# Live feed variables
ENABLE_DISPLAY = True
WINDOW_NAME = 'PathPal Live'
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# models path
COCO_SSD_MOBILENET_V1_PATH = 'models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'
COCO_LABELS_PATH = 'models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt'

# camera settings
CAM_WIDTH = 320 * 2
CAM_HEIGHT = 240 * 2
FPS = 15

# runtime settings
INTERPRETER_MODE = 'runtime'

# Stream settings
ENABLE_STREAM = False
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 70
STREAM_FPS = 5.0

# confidence threshold
COCO_DEFAULT_THRESH = 0.45
COCO_THRESHOLDS = {
    'person': 0.55,
    'bottle': 0.2,

}
DEBUG_TOP_N = 10

# intrested labels
WANTED_LABELS = {
    'person',
    'car',
    'bus',
    'truck',
    'bicycle',
    'motorcycle',
    'bottle',
    'cell phone'
}