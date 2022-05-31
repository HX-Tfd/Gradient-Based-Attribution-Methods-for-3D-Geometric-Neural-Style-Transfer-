from enum import Enum

"""
constants for the entire dataset
"""
NUM_SHAPE_CATEGORIES = 16
NUM_PART_CATEGORIES = 50
MAX_PARTS = 6

class ShapeId(Enum):
    AIRPLANE = 0
    BAG = 1
    CAP = 2
    CAR = 3
    CHAIR = 4
    EARPHONE = 5
    GUITAR = 6
    KNIFE = 7
    LAMP = 8
    LAPTOP = 9
    MOTORBIKE = 10
    MUG = 11
    PISTOL = 12
    ROCKET = 13
    SKATEBOARD = 14
    TABLE = 15


"""
constants used for the experimental dataset (subset of ShapeNet)
"""
#TODO: maybe implement this as a separate config file
CLASS_LIST = [ 'bag', 'bench', 'chair', 'guitar',  'lamp', 'sofa', 'table']
NUM_SHAPE_CATEGORIES_PARTIAL = len(CLASS_LIST)
