import cv2
import numpy as np


# ----------------------------------- CONSTANT ----------------------------------- #
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BIRD_WIDTH = 75
BIRD_HEIGHT = 60
PIPE_WIDTH = 80
PIPE_HEIGHT = 300
PIPE_GAP = 200
PIPE_SPEED = 5
GRAVITY = 1
JUMP_VELOCITY = 1
PIPE_NUMBER = 5
COLORS = [
    (212, 212, 255),
    (144, 233, 205),
    (115, 203, 170),
    (185, 152, 185),
    (196, 213, 232)
]

NOCOLOR = "\033[0m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
ORANGE = "\033[0;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
LIGHTGRAY = "\033[0;37m"
DARKGRAY = "\033[1;30m"
LIGHTRED = "\033[1;31m"
LIGHTGREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
LIGHTBLUE = "\033[1;34m"
LIGHTPURPLE = "\033[1;35m"
LIGHTCYAN = "\033[1;36m"
WHITE = "\033[1;37m"

# ----------------------------------- FUNCTIONS ----------------------------------- #

def read_bgra(path, width=None, height=None):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED) if type(path) is str else path
    if width is not None and height is not None:
        image = cv2.resize(image, (width, height))
    mask = image[:, :, 3] / 255.
    mask = cv2.merge([mask, mask, mask])
    image = image[:, :, :3]
    return image, mask


def rollback(image, mask, target, x1, y1, x2, y2):
    final = np.uint32(target * mask + image[y1: y2, x1: x2] * (1 - mask))
    image[y1: y2, x1: x2] = final
    return image

