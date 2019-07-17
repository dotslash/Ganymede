import json
import os
import random
from glob import glob
from os import path
from typing import Dict, Tuple, List, Set, Callable, Iterator

import attr
import numpy as np
import pandas
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, Callback
from keras.engine import InputLayer
from keras.layers import Concatenate, GlobalMaxPool2D, Subtract, Multiply, \
    Dense, Dropout, GlobalAvgPool2D
from keras.optimizers import Adam
from keras_preprocessing import image
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input

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
    negative_sample_ratio: float = attr.attrib(default=0.5)
    batch_size: int = attr.attrib(default=16)
    # update_params overrides this in test_mode
    epochs: int = attr.attrib(default=50)
    # update_params overrides this in test_mode
    steps_per_epoch: int = attr.attrib(default=200)
    # update_params overrides this in test_mode
    validation_steps: int = attr.attrib(default=100)
    optimizer_lr: float = attr.attrib(default=0.00001)
    k_fold_count: int = attr.attrib(default=5)
    dropout: float = attr.ib(default=0.2)
    image_size: int = attr.ib(default=224)

    def get_inp_path(self, file_name):
        return self.input_data_dir + file_name

    def get_out_path(self, file_name):
        return self.output_data_dir + file_name

    def update_params(self, config_file_path):
        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)
                print("Before update_params:", self)
                for k, v in config.items():
                    if k.endswith('_dir'):
                        v = full_dir_path(v)
                    self.__dict__[k] = v
            print("After update_params:", self)
        except Exception as e:
            print("Exception while loading config.json. This is okay when "
                  "running on kaggle. Error:", e)
        if self.test_mode:
            self.epochs = min(self.epochs, 5)
            self.validation_steps = min(self.validation_steps, 5)
            self.steps_per_epoch = min(self.steps_per_epoch, 2)
        return self


@attr.s
class PeopleTrainValSplit:
    train_people: List[PersonId] = attr.ib(default=[])
    train_relations: List[Tuple[PersonId, PersonId]] = attr.ib(default=[])
    val_people: List[PersonId] = attr.ib(default=[])
    val_relations: List[Tuple[PersonId, PersonId]] = attr.ib(default=[])


class Data:
    @staticmethod
    def load_image_as_array(filepath: str, params: Params) -> np.ndarray:
        img = image.load_img(filepath,
                             target_size=(params.image_size, params.image_size))
        img = np.array(img).astype(np.float)
        return preprocess_input(img, version=2)

    @staticmethod
    def _load_train_images(params: Params) -> Dict[PersonId, List[str]]:
        ret: Dict[PersonId, List[str]] = {}
        all_images = glob(params.get_inp_path('train/') + "*/*/*.jpg")
        for name in all_images:
            # img_np = Data._load_image_as_array(name, params)
            p: PersonId = (name.split('/')[-3], name.split('/')[-2])
            if p not in ret:
                ret[p] = []
            ret[p].append(name)
            if params.test_mode and len(ret) > 200:
                break
        print("Train+Val images: ", sum(len(x) for x in ret.values()))
        print("Train+Val people: ", len(ret))
        return ret

    @staticmethod
    def _load_test_images(params: Params) -> Dict[str, str]:
        ret: Dict[str, str] = {}
        all_images = glob(params.get_inp_path('test/') + "*.jpg")
        for name in all_images:
            # img_np = Data._load_image_as_array(name, params)
            ret[name.split('/')[-1]] = name
            if params.test_mode and len(ret) > 500:
                break
        print("Test images: ", len(ret))
        return ret

    @staticmethod
    def _train_or_val_batch_generator(
            params: Params, images: Dict[PersonId, List[str]],
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
                p1img, p2img = random.choice(images[p1]), random.choice(
                    images[p2])
                p1img, p2img = Data.load_image_as_array(p1img, params), \
                               Data.load_image_as_array(p2img, params)
                x1.append(p1img)
                x2.append(p2img)
                out.append(1)
            while len(x1) < positives + negatives:
                p1 = random.choice(people)
                p2 = random.choice(people)
                if (p1, p2) in relations or (p2, p1) in relations:
                    continue
                p1img, p2img = random.choice(images[p1]), random.choice(
                    images[p2])
                p1img, p2img = Data.load_image_as_array(p1img, params), \
                               Data.load_image_as_array(p2img, params)
                x1.append(p1img)
                x2.append(p2img)
                out.append(0)
            yield [np.array(x1), np.array(x2)], out

    def __init__(self, params: Params):
        self.params = params
        self.test_images: Dict[str, str] = Data._load_test_images(
            self.params)
        print("d2")
        self.images: Dict[
            PersonId, List[str]] = Data._load_train_images(self.params)
        print("d1")

        self.kfold_split = self.k_fold_split_people_and_relations(
            params.k_fold_count)
        print("d3")

        # Generate test data.
        # [img_par], [p1_image], [p2_image]
        self.test_data: List[Tuple[str, str, str]] = []
        test_img_pairs = pandas.read_csv(
            self.params.get_inp_path('sample_submission.csv'))['img_pair']
        failed_image_pairs = []
        for img_pair in test_img_pairs:
            img1, img2 = img_pair.split('-')
            img1, img2 = self.test_images.get(img1), self.test_images.get(img2)
            if img1 is not None and img2 is not None:
                self.test_data.append((img1, img2, img_pair))
            else:
                failed_image_pairs.append(img_pair)
        if failed_image_pairs:
            print("Failed to find {} image pairs".format(
                len(failed_image_pairs)))
            print("pairs: ", ','.join(failed_image_pairs)[:500], " ...")

    def k_fold_split_people_and_relations(self, num_folds=5) \
            -> List[PeopleTrainValSplit]:
        relations_df: pandas.DataFrame = pandas.read_csv(
            self.params.get_inp_path('train_relationships.csv'))
        get_person_id: Callable[[str], PersonId] = lambda x: (
            x.split('/')[0], x.split('/')[1])
        relations: List[Tuple[PersonId, PersonId]] = list(
            zip(list(relations_df["p1"].apply(get_person_id)),
                list(relations_df["p1"].apply(get_person_id))))
        if not self.params.test_mode:
            random.shuffle(relations)
        _families = set(x[0][0] for x in relations)
        _families.update(set(x[1][0] for x in relations))
        _families = list(_families)
        ret = []
        for i in range(num_folds):
            split: PeopleTrainValSplit = PeopleTrainValSplit()
            ret.append(split)
            val_family_range_start = (len(_families) * i) // num_folds
            # Dont allow more than 20% of data into validation set.
            # TODO(dotslash): Is this needed?
            val_family_range_end = val_family_range_start + min(
                len(_families) // 5, len(_families) // num_folds)
            val_families: Set[str] = set(
                _families[val_family_range_start:val_family_range_end])
            for relation in relations:
                if relation[0][0] in val_families or \
                        relation[1][0] in val_families:
                    split.val_relations.append(relation)
                else:
                    split.train_relations.append(relation)
            # Split people into train and validation.
            for person in self.images.keys():
                if person[0] in val_families:
                    split.val_people.append(person)
                else:
                    split.train_people.append(person)
        return ret

    def get_test_data(self) -> List[Tuple[str, str, str]]:
        return self.test_data

    def val_batch_generator(self, fold_num) -> Iterator[TrainValDataType]:
        split: PeopleTrainValSplit = self.kfold_split[fold_num]
        return Data._train_or_val_batch_generator(self.params, self.images,
                                                  split.val_relations,
                                                  split.val_people)

    def train_batch_generator(self, fold_num) -> Iterator[TrainValDataType]:
        split: PeopleTrainValSplit = self.kfold_split[fold_num]
        return Data._train_or_val_batch_generator(self.params, self.images,
                                                  split.train_relations,
                                                  split.train_people)

    def print_summary(self):
        # print('Test data info: i1-shape: {} i2-shape: {}'.format(
        #     np.stack(self.test_data[0]).shape,
        #     np.stack(self.test_data[1]).shape))
        # for X, y in self.train_batch_generator(0):
        #     print('Train batch shapes: x1: {} x2: {} y: {}'.format(
        #         X[0].shape, X[1].shape, len(y)))
        #     break
        # for X, y in self.val_batch_generator(0):
        #     print('Val batch shapes: x1: {} x2: {} y: {}'.format(
        #         X[0].shape, X[1].shape, len(y)))
        #     break
        pass


def create_model(params: Params) -> Model:
    input1: InputLayer = Input(shape=(params.image_size, params.image_size, 3))
    input2: InputLayer = Input(shape=(params.image_size, params.image_size, 3))

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
    x = Dropout(params.dropout)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([input1, input2], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'],
                  optimizer=Adam(params.optimizer_lr))
    return model


class KaggleLoggingCB(Callback):
    def __init__(self, model_ind):
        super().__init__()
        self.cur_epoch: int = 0
        self.model_ind = model_ind

    def on_epoch_begin(self, epoch, logs=None):
        self.cur_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        message = 'model_{}: epoch {}'.format(self.model_ind, epoch)
        os.system('echo {}'.format(message))
        print(message)

    def on_batch_end(self, batch, logs=None):
        if batch % 50 != 0:
            return
        message = 'model_{}: Training {}.{} ends'.format(self.model_ind,
                                                         self.cur_epoch, batch)
        os.system('echo {}'.format(message))
        print(message)


def main():
    params: Params = Params().update_params('./config.json')
    os.system('echo loaded params')
    print('echo loaded params')

    data: Data = Data(params)
    data.print_summary()
    os.system('echo Loaded data')
    print('echo Loaded data')

    predictions = [0] * len(data.test_data)
    for i in range(params.k_fold_count):
        model = create_model(params)
        if i == 0:
            model.summary()
        os.system('echo Created model')
        print('echo Created model')
        control_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=10,
                                       verbose=1)
        model.fit_generator(generator=data.train_batch_generator(i),
                            validation_data=data.val_batch_generator(i),
                            steps_per_epoch=params.steps_per_epoch,
                            validation_steps=params.validation_steps,
                            epochs=params.epochs, verbose=2,
                            use_multiprocessing=True, workers=4,
                            callbacks=[control_lr, KaggleLoggingCB(i)])
        model.save(params.get_out_path("kinship_{}.m5".format(i)))
        for ind, (i1, i2, _) in enumerate(data.test_data):
            x1 = [Data.load_image_as_array(i1, params)]
            x2 = [Data.load_image_as_array(i2, params)]
            prediction = model.predict([x1, x2])[0][0]
            predictions[ind] += (prediction / params.k_fold_count)

    submission = pandas.DataFrame(
        data={'is_related': predictions,
              'img_pair': [x[2] for x in data.test_data]})
    submission.to_csv(params.get_out_path("output.csv"), index=False)


if __name__ == '__main__':
    main()
