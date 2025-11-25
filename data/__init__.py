from data.data_stats import BIRD_STATS, DOG_STATS, FLOWER_STATS, PET_STATS, CAR_STATS, POKEMON_STATS, AIRCRAFT_STATS, EUROSAT_STATS, FOOD_STATS, DTD_STATS, CALTECH101_STATS, CALTECH256_STATS, DEEPFASHION_MULTIMODAL_STATS, SUN397_STATS, IMAGENET_A_STATS, IMAGENET_R_STATS
from data.bird200 import build_bird_prompter, build_bird200_discovery, build_bird200_test, bird_how_to1, bird_how_to2, build_bird200_swav_train
from data.dog120 import build_dog_prompter, build_dog120_discovery, build_dog120_test, dog_how_to1, dog_how_to2, build_dog120_swav_train
from data.flower102 import build_flower_prompter, build_flower102_discovery, build_flower102_test, flower_how_to1, flower_how_to2, build_flower102_swav_train
from data.pet37 import build_pet_prompter, build_pet37_discovery, build_pet37_test, pet_how_to1, pet_how_to2, build_pet37_swav_train
from data.car196 import build_car_prompter, build_car196_discovery, build_car196_test, car_how_to1, car_how_to2, build_car196_swav_train
from data.fgvc_aircraft import build_aircraft_prompter, build_aircraft100_discovery, build_aircraft100_test, aircraft_how_to1, aircraft_how_to2, build_aircraft100_swav_train
from data.eurosat import build_eurosat_prompter, build_eurosat10_discovery, build_eurosat10_test, eurosat_how_to1, eurosat_how_to2, build_eurosat10_swav_train
from data.food_101 import build_food_prompter, build_food101_discovery, build_food101_test, food_how_to1, food_how_to2, build_food101_swav_train
from data.dtd import build_dtd_prompter, build_dtd47_discovery, build_dtd47_test, dtd_how_to1, dtd_how_to2, build_dtd47_swav_train
from data.caltech101 import build_caltech101_prompter, build_caltech101_discovery, build_caltech101_test, caltech101_how_to1, caltech101_how_to2, build_caltech101_swav_train
from data.caltech256 import build_caltech256_prompter, build_caltech256_discovery, build_caltech256_test, caltech256_how_to1, caltech256_how_to2, build_caltech256_swav_train
from data.deepfashion_multimodal import build_deepfashion_multimodal_prompter, build_deepfashion_multimodal_discovery, build_deepfashion_multimodal_test, deepfashion_multimodal_how_to1, deepfashion_multimodal_how_to2, build_deepfashion_multimodal_swav_train
from data.sun397 import build_sun397_prompter, build_sun397_discovery, build_sun397_test, sun397_how_to1, sun397_how_to2, build_sun397_swav_train
from data.imagenet_a import build_imagenet_a_prompter, build_imagenet_a_discovery, build_imagenet_a_test, imagenet_a_how_to1, imagenet_a_how_to2, build_imagenet_a_swav_train
from data.imagenet_r import build_imagenet_r_prompter, build_imagenet_r_discovery, build_imagenet_r_test, imagenet_r_how_to1, imagenet_r_how_to2, build_imagenet_r_swav_train

from data.bird200 import _transform as bird_transform
from data.car196 import _transform as car_transform
from data.dog120 import _transform as dog_transform
from data.flower102 import _transform as flower_transform
from data.pet37 import _transform as pet_transform
from data.fgvc_aircraft import _transform as aircraft_transform
from data.eurosat import _transform as eurosat_transform
from data.food_101 import _transform as food_transform
from data.dtd import _transform as dtd_transform
from data.caltech101 import _transform as caltech101_transform
from data.caltech256 import _transform as caltech256_transform
from data.deepfashion_multimodal import _transform as deepfashion_multimodal_transform
from data.sun397 import _transform as sun397_transform
from data.imagenet_a import _transform as imagenet_a_transform
from data.imagenet_r import _transform as imagenet_r_transform

from .utils import random_augmentation

__all__ = [
    "DATA_STATS", "PROMPTERS", "DATA_DISCOVERY", "DATA_GROUPING", "DATA_TRANSFORM",
    "BIRD_STATS", "DOG_STATS", "FLOWER_STATS", "PET_STATS", "CAR_STATS", "POKEMON_STATS",
    "build_bird_prompter", "build_bird200_discovery", "build_bird200_test",
    "build_dog_prompter", "build_dog120_discovery", "build_dog120_test",
    "build_flower_prompter", "build_flower102_discovery", "build_flower102_test",
    "build_pet_prompter", "build_pet37_discovery", "build_pet37_test",
    "build_car_prompter", "build_car196_discovery", "build_car196_test",
    "build_pokemon_prompter", "build_pokemon_discovery", "build_pokemon_test"
]


HOW_TOS1 = {
    "bird": bird_how_to1,
    "dog": dog_how_to1,
    "flower": flower_how_to1,
    "pet": pet_how_to1,
    "car": car_how_to1,
    "aircraft": aircraft_how_to1,
    "eurosat": eurosat_how_to1,
    "food": food_how_to1,
    "dtd": dtd_how_to1,
    "caltech101": caltech101_how_to1,
    "caltech256": caltech256_how_to1,
    "deepfashion_multimodal": deepfashion_multimodal_how_to1,
    "sun397": sun397_how_to1,
    "imagenet_a": imagenet_a_how_to1,
    "imagenet_r": imagenet_r_how_to1,
}

HOW_TOS2 = {
    "bird": bird_how_to2,
    "dog": dog_how_to2,
    "flower": flower_how_to2,
    "pet": pet_how_to2,
    "car": car_how_to2,
    "aircraft": aircraft_how_to2,
    "eurosat": eurosat_how_to2,
    "food": food_how_to2,
    "dtd": dtd_how_to2,
    "caltech101": caltech101_how_to2,
    "caltech256": caltech256_how_to2,
    "deepfashion_multimodal": deepfashion_multimodal_how_to2,
    "sun397": sun397_how_to2,
    "imagenet_a": imagenet_a_how_to2,
    "imagenet_r": imagenet_r_how_to2,
}

DATA_STATS = {
    "bird": BIRD_STATS,
    "dog": DOG_STATS,
    "flower": FLOWER_STATS,
    "pet": PET_STATS,
    "car": CAR_STATS,
    "aircraft": AIRCRAFT_STATS,
    "eurosat": EUROSAT_STATS,
    "food": FOOD_STATS,
    "dtd": DTD_STATS,
    "caltech101": CALTECH101_STATS,
    "caltech256": CALTECH256_STATS,
    "deepfashion_multimodal": DEEPFASHION_MULTIMODAL_STATS,
    "sun397": SUN397_STATS,
    "imagenet_a": IMAGENET_A_STATS,
    "imagenet_r": IMAGENET_R_STATS,
}


PROMPTERS = {
    "bird": build_bird_prompter,
    "dog": build_dog_prompter,
    "flower": build_flower_prompter,
    "pet": build_pet_prompter,
    "car": build_car_prompter,
    "aircraft": build_aircraft_prompter,
    "eurosat": build_eurosat_prompter,
    "food": build_food_prompter,
    "dtd": build_dtd_prompter,
    "caltech101": build_caltech101_prompter,
    "caltech256": build_caltech256_prompter,
    "deepfashion_multimodal": build_deepfashion_multimodal_prompter,
    "sun397": build_sun397_prompter,
    "imagenet_a": build_imagenet_a_prompter,
    "imagenet_r": build_imagenet_r_prompter,
}


DATA_DISCOVERY = {
    "bird": build_bird200_discovery,
    "dog": build_dog120_discovery,
    "flower": build_flower102_discovery,
    "pet": build_pet37_discovery,
    "car": build_car196_discovery,
    "aircraft": build_aircraft100_discovery,
    "eurosat": build_eurosat10_discovery,
    "food": build_food101_discovery,
    "dtd": build_dtd47_discovery,
    "caltech101": build_caltech101_discovery,
    "caltech256": build_caltech256_discovery,
    "deepfashion_multimodal": build_deepfashion_multimodal_discovery,
    "sun397": build_sun397_discovery,
    "imagenet_a": build_imagenet_a_discovery,
    "imagenet_r": build_imagenet_r_discovery,
}


DATA_GROUPING = {
    "bird": build_bird200_test,
    "dog": build_dog120_test,
    "flower": build_flower102_test,
    "pet": build_pet37_test,
    "car": build_car196_test,
    "aircraft": build_aircraft100_test,
    "eurosat": build_eurosat10_test,
    "food": build_food101_test,
    "dtd": build_dtd47_test,
    "caltech101": build_caltech101_test,
    "caltech256": build_caltech256_test,
    "deepfashion_multimodal": build_deepfashion_multimodal_test,
    "sun397": build_sun397_test,
    "imagenet_a": build_imagenet_a_test,
    "imagenet_r": build_imagenet_r_test,
}


DATA_TRANSFORM = {
    "bird": bird_transform,
    "dog": dog_transform,
    "flower": flower_transform,
    "pet": pet_transform,
    "car": car_transform,
    "aircraft": aircraft_transform,
    "eurosat": eurosat_transform,
    "food": food_transform,
    "dtd": dtd_transform,
    "caltech101": caltech101_transform,
    "caltech256": caltech256_transform,
    "deepfashion_multimodal": deepfashion_multimodal_transform,
    "sun397": sun397_transform,
    "imagenet_a": imagenet_a_transform,
    "imagenet_r": imagenet_r_transform,
}


DATA_SWAV = {
    "bird": build_bird200_swav_train,
    "dog": build_dog120_swav_train,
    "flower": build_flower102_swav_train,
    "pet": build_pet37_swav_train,
    "car": build_car196_swav_train,
    "aircraft": build_aircraft100_swav_train,
    "eurosat": build_eurosat10_swav_train,
    "food": build_food101_swav_train,
    "dtd": build_dtd47_swav_train,
    "caltech101": build_caltech101_swav_train,
    "caltech256": build_caltech256_swav_train,
    "deepfashion_multimodal": build_deepfashion_multimodal_swav_train,
    "sun397": build_sun397_swav_train,
    "imagenet_a": build_imagenet_a_swav_train,
    "imagenet_r": build_imagenet_r_swav_train,
}

DATA_AUGMENTATION = random_augmentation


