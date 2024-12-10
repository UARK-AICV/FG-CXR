from . import data  # register all new datasets
from . import model
from . import utils
from . import evaluator
# config
from .config import add_iai_config

# dataset loading

from .data.dataset_mappers.IAI_gaze_dataset_mapper_e2e_v2 import IAIGazeDatasetMapper_e2e_v2
# models
from .test_time_augmentation import SemanticSegmentorWithTTA
