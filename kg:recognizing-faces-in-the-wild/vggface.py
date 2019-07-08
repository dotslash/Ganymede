import json
import random
from glob import glob
from os import path
from typing import Dict, Tuple, List, Set, Callable, Iterator

import attr
import cv2
import numpy as np
import pandas
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.engine import InputLayer
from keras.layers import Concatenate, GlobalMaxPool2D, Subtract, Multiply, \
    Dense, Dropout, GlobalAvgPool2D
from keras.optimizers import Adam
from keras_vggface import VGGFace

PersonId = Tuple[str, str]
TrainValDataType = Iterator[Tuple[List[np.ndarray], List[int]]]
TestDataType = Tuple[List[np.ndarray], List[np.ndarray], List[str]]


def full_dir_path(relative_path: str) -> str:
    ret = path.abspath(path.expanduser(relative_path))
    if not ret.endswith('/'):
        ret = ret + '/'
    return ret


@attr.s
class Params(object):
    # Test mode runs quickly and produces horrible results!
    test_mode: float = attr.attrib(default=False)

    families_val_split: float = attr.attrib(default=0.1)
    input_data_dir: str = attr.attrib(default=full_dir_path('../input/'))
    output_data_dir: str = attr.attrib(default=full_dir_path('./'))
    # 1 +ve sample, 1 -ve sample.
    negative_sample_ratio: float = attr.attrib(default=0.5)
    # Batches of 32 (16 +ve, 16 -ve)
    batch_size: int = attr.attrib(default=32)
    # initiate_config overrides this in test_mode
    epochs: int = attr.attrib(default=100)
    # initiate_config overrides this in test_mode
    steps_per_epoch: int = attr.attrib(default=200)
    # initiate_config overrides this in test_mode
    validation_steps: int = attr.attrib(default=100)

    def get_inp_path(self, file_name):
        return self.input_data_dir + file_name

    def get_out_path(self, file_name):
        return self.output_data_dir + file_name

    def initiate_config(self):
        try:
            config_path = path.abspath(path.dirname(__file__) + '/config.json')
            with open(config_path, "r") as f:
                config = json.load(f)
                print("Before initiate_config:", self)
                for k, v in config.items():
                    if k.endswith('_dir'):
                        v = full_dir_path(v)
                    self.__dict__[k] = v
            print("After initiate_config:", self)
        except Exception as e:
            print("Exception while loading config.json. This is okay when "
                  "running on kaggle. Error:", e)
        if self.test_mode:
            self.epochs = min(self.epochs, 5)
            self.validation_steps = min(self.validation_steps, 5)
            self.steps_per_epoch = min(self.steps_per_epoch, 2)


class Data:
    @staticmethod
    def _load_image_as_array(filepath: str) -> np.ndarray:
        with open(filepath, 'rb') as f:
            # noinspection PyTypeChecker
            nparr = np.fromstring(f.read(), np.uint8)
            # noinspection PyUnresolvedReferences
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def _load_train_images(params: Params) -> Dict[PersonId, List[np.ndarray]]:
        ret: Dict[PersonId, List[np.ndarray]] = {}
        all_images = glob(params.get_inp_path('train/') + "*/*/*.jpg")
        for name in all_images:
            img_np = Data._load_image_as_array(name)
            p: PersonId = (name.split('/')[-3], name.split('/')[-2])
            if p not in ret:
                ret[p] = []
            ret[p].append(img_np)
            if params.test_mode and len(ret) > 200:
                break
        print("Train+Val images: ", sum(len(x) for x in ret.values()))
        print("Train+Val people: ", len(ret))
        return ret

    @staticmethod
    def _load_test_images(params: Params) -> Dict[str, np.ndarray]:
        ret: Dict[str, np.ndarray] = {}
        all_images = glob(params.get_inp_path('test/') + "*.jpg")
        for name in all_images:
            img_np = Data._load_image_as_array(name)
            ret[name.split('/')[-1]] = img_np
            if params.test_mode and len(ret) > 500:
                break
        print("Test images: ", len(ret))
        return ret

    @staticmethod
    def _train_or_val_batch_generator(
            params: Params, images: Dict[PersonId, List[np.ndarray]],
            relations: List[Tuple[PersonId, PersonId]],
            people: List[PersonId]) -> \
            Iterator[TrainValDataType]:
        negatives = int(
            params.batch_size * params.negative_sample_ratio)
        positives = params.batch_size - negatives
        while True:
            x1: List[np.ndarray] = []
            x2: List[np.ndarray] = []
            out: List[int] = []
            while len(x1) < positives:
                p1, p2 = random.choice(relations)
                if p1 not in images or p2 not in images:
                    continue
                p1img, p2img = random.choice(images[p1]), \
                               random.choice(images[p2])
                x1.append(p1img)
                x2.append(p2img)
                out.append(1)
            while len(x1) < positives + negatives:
                p1 = random.choice(people)
                p2 = random.choice(people)
                if (p1, p2) in relations or (p2, p1) in relations:
                    continue
                p1img, p2img = random.choice(images[p1]), \
                               random.choice(images[p2])
                x1.append(p1img)
                x2.append(p2img)
                out.append(0)
            yield [np.array(x1), np.array(x2)], out

    def __init__(self, params: Params):
        self.params = params
        self.images: Dict[
            PersonId, List[np.ndarray]] = Data._load_train_images(self.params)
        self.test_images: Dict[str, np.ndarray] = Data._load_test_images(
            self.params)

        # Generate validation relations and train relations.
        # noinspection PyTypeChecker
        self.val_relations: List[Tuple[PersonId, PersonId]] = []
        # noinspection PyTypeChecker
        self.train_relations: List[Tuple[PersonId, PersonId]] = []
        get_person_id: Callable[[str], PersonId] = \
            lambda x: (x.split('/')[0], x.split('/')[1])
        relations_df: pandas.DataFrame = pandas.read_csv(
            self.params.get_inp_path('train_relationships.csv'))
        relations: List[Tuple[PersonId, PersonId]] = \
            list(zip(list(relations_df["p1"].apply(get_person_id)),
                     list(relations_df["p1"].apply(get_person_id))))
        if not params.test_mode:
            random.shuffle(relations)
        _families = set(x[0][0] for x in relations)
        _families.update(set(x[1][0] for x in relations))
        _families = list(_families)
        val_families: Set[str] = set(_families[:int(
            len(_families) * self.params.families_val_split)])
        for relation in relations:
            if relation[0][0] in val_families or relation[1][0] in val_families:
                self.val_relations.append(relation)
            else:
                self.train_relations.append(relation)
        # Split people into train and validation.
        self.train_people: List[PersonId] = []
        self.val_people: List[PersonId] = []
        for person in self.images.keys():
            if person[0] in val_families:
                self.val_people.append(person)
            else:
                self.train_people.append(person)

        # Generate test data.
        # [img_par], [p1_image], [p2_image]
        self.test_data: TestDataType = ([], [], [])
        test_img_pairs = pandas.read_csv(
            self.params.get_inp_path('sample_submission.csv'))['img_pair']
        failed_image_pairs = []
        for img_pair in test_img_pairs:
            img1, img2 = img_pair.split('-')
            img1, img2 = self.test_images.get(img1), self.test_images.get(img2)
            if img1 is not None and img2 is not None:
                self.test_data[0].append(img1)
                self.test_data[1].append(img2)
                self.test_data[2].append(img_pair)
            else:
                failed_image_pairs.append(img_pair)
        if failed_image_pairs:
            print("Failed to find {} image pairs".format(
                len(failed_image_pairs)))
            print("pairs: ", ','.join(failed_image_pairs)[:500], " ...")

    def get_test_data(self) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        return self.test_data

    def val_batch_generator(self) -> Iterator[TrainValDataType]:
        return Data._train_or_val_batch_generator(self.params, self.images,
                                                  self.val_relations,
                                                  self.val_people)

    def train_batch_generator(self) -> Iterator[TrainValDataType]:
        return Data._train_or_val_batch_generator(self.params, self.images,
                                                  self.train_relations,
                                                  self.train_people)

    def print_summary(self):
        print('Test data info: i1-shape: {} i2-shape: {}'.format(
            np.stack(self.test_data[0]).shape,
            np.stack(self.test_data[1]).shape))
        for X, y in self.train_batch_generator():
            print('Train batch shapes: x1: {} x2: {} y: {}'.format(
                X[0].shape, X[1].shape, len(y)))
            break
        for X, y in self.val_batch_generator():
            print('Val batch shapes: x1: {} x2: {} y: {}'.format(
                X[0].shape, X[1].shape, len(y)))
            break


def create_model() -> Model:
    input1: InputLayer = Input(shape=(224, 224, 3))
    input2: InputLayer = Input(shape=(224, 224, 3))

    base_model: Model = VGGFace(model='resnet50', include_top=False)

    # Make last 3 layers trainable.
    for x in base_model.layers[:-3]:
        x.trainable = True

    # Transform image1
    x1 = base_model(input1)
    x1 = Concatenate()([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])

    # Transform image2
    x2 = base_model(input2)
    x2 = Concatenate()([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    _diff = Subtract()([x1, x2])
    diff_squared = Multiply()([_diff, _diff])

    # concat(x1.x2, (x1-x2)**2)
    x = Concatenate()([Multiply()([x1, x2]), diff_squared])
    x = Dense(100, activation="relu")(x)
    # TODO(dotslash): Not sure about the dropout prob.
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([input1, input2], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'],
                  optimizer=Adam(0.00001))
    model.summary()
    return model


def main() -> None:
    params: Params = Params()
    params.initiate_config()
    data: Data = Data(params)
    data.print_summary()
    model = create_model()
    save_best_model = ModelCheckpoint(params.get_out_path('best_model.h5'),
                                      monitor='val_acc', verbose=1,
                                      save_best_only=True)
    control_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    stop_early = EarlyStopping(monitor='val_loss', patience=20)
    model.fit_generator(generator=data.train_batch_generator(),
                        validation_data=data.val_batch_generator(),
                        steps_per_epoch=params.steps_per_epoch,
                        validation_steps=params.validation_steps,
                        epochs=params.epochs,
                        callbacks=[save_best_model, control_lr, stop_early])
    predictions = model.predict_on_batch(
        x=[data.test_data[0], data.test_data[1]])
    submission = pandas.DataFrame(
        data={'is_related': predictions.flatten(),
              'img_pair': data.test_data[2]})
    submission.to_csv(params.get_out_path("output.csv"), index=False)


if __name__ == '__main__':
    main()
