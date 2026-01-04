# Live feed variables
ENABLE_DISPLAY = True
WINDOW_NAME = 'PathPal Live'
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# models path
COCO_SSD_MOBILENET_V1_PATH = 'models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite'
COCO_SSD_MOBILENET_V3_LARGE_PATH = 'models/ssd_mobilenet_v3_large_coco_2020_01_14/model.tflite'
EFFICIENTDET_V0_PATH = 'models/efficientdet/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
EFFICIENTDET_V1_PATH = 'models/efficientdet/lite-model_efficientdet_lite1_detection_metadata_1.tflite'
EFFICIENTDET_V2_PATH = 'models/efficientdet/lite-model_efficientdet_lite2_detection_metadata_1.tflite'
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
    'mouse': 0.2,

}
DEBUG_TOP_N = 10
COCO_NMS_THRESH = 0.45

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

# debug statements
DEBUG = True


# Ultrasonic Sensor
ENABLE_ULTRASONIC = False
TRIGGER = 4 # GPIO4
ECHO = 17 # GPIO17
MAX_DISTANCE = 10 # in metres
RANGE_SMOOTH_N = 5
TARGET_DIRECTION = 'center' # checks objects in central region of cam
# Which labels are allowed to drive the "direction" (priority order)
RANGE_DIR_PRIORITY = ['person']          # e.g. ["person", "bicycle", "car"]
RANGE_DIR_FALLBACK = 'largest_any'       # "largest_any" | "center_only" | "none"

OBSTACLE_NEAR_CM = 80.0
OBSTACLE_FAR_CM  = 250.0
'''
If you want the direction to be based on person first, then cars, then bikes:

RANGE_DIR_PRIORITY = ["person", "car", "bicycle"]
RANGE_DIR_FALLBACK = "largest_any"


If you never want vision to decide direction (only “ahead”):

RANGE_DIR_FALLBACK = "center_only"
RANGE_DIR_PRIORITY = []
'''

# fps stats
ENABLE_FPS = True
TARGET_FRAME_GRABBER_FPS = 30


# TFLITE
TFLITE_THREADS = 3