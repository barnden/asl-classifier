import lmdb
import os
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from random import random

def preprocess_image(directory, color_file, target_size=(64, 64)):
    # depth_file = "depth" + color_file[5:]
    img = Image.open(os.path.join(directory, color_file))
    # depth = Image.open(os.path.join(directory, depth_file))

    img = img.resize(target_size)
    # depth = depth.resize(target_size)

    # threshold = img.convert("L")
    # threshold = img.filter(ImageFilter.FIND_EDGES)
    # threshold = np.array(threshold, dtype=np.float64)[..., :1] / 255

    img = np.array(img, dtype=np.float32) / 255
    # depth = np.array(depth, dtype=np.float64)

    # depth -= depth.min()
    # depth /= depth.max()

    # depth = depth[..., None]

    # img = np.concatenate((img, depth, threshold), axis=-1)

    return img

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

if __name__ == "__main__":
    env = lmdb.open("asl_64x64.lmdb", map_size=1099511627776)

    train_count = 0
    validate_count = 0
    test_count = 0

    with env.begin(write=True) as txn:
        for group in "ABCDE":
            group_path = os.path.join("./dataset5", group)

            for directory in os.listdir(group_path):
                subdirectories = filter(lambda x: not os.path.isfile(x), directory)

                for subdirectory in subdirectories:
                    subdirectory_path = os.path.join(group_path, subdirectory)

                    images = filter(lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(subdirectory_path))
                    color_images = filter(lambda x: x.lower().startswith('color'), images)

                    for image in color_images:
                        preprocessed_image = preprocess_image(subdirectory_path, image)
                        label = alphabet[int(image.split('_')[1])]

                        if group == 'E':
                            txn.put(f'test:image:{test_count}'.encode(), preprocessed_image.tobytes())
                            txn.put(f'test:label:{test_count}'.encode(), label.encode())
                            test_count += 1
                        elif random() < .85:
                            txn.put(f'train:image:{train_count}'.encode(), preprocessed_image.tobytes())
                            txn.put(f'train:label:{train_count}'.encode(), label.encode())
                            train_count += 1
                        else:
                            txn.put(f'validate:image:{validate_count}'.encode(), preprocessed_image.tobytes())
                            txn.put(f'validate:label:{validate_count}'.encode(), label.encode())
                            validate_count += 1

            print(f"train {train_count} validate {validate_count} test {test_count}")

        for group in ["asl_alphabet_train", "Test_Alphabet", "Train_Alphabet"]:
            # for directory in os.listdir(f"./{group}"):
            for subdirectory in "ABCDEFGHIKLMNOPQRSTUVWXY":
                subdirectory_path = os.path.join("./", group, subdirectory)
                images = filter(lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(subdirectory_path))

                label = subdirectory.encode()
                for image in images:
                    preprocessed_image = preprocess_image(subdirectory_path, image)

                    if group == "asl_alphabet_test" or group == "Test_Alphabet":
                        txn.put(f'test:image:{test_count}'.encode(), preprocessed_image.tobytes())
                        txn.put(f'test:label:{test_count}'.encode(), label)
                        test_count += 1
                    elif random() < .85:
                        txn.put(f'train:image:{train_count}'.encode(), preprocessed_image.tobytes())
                        txn.put(f'train:label:{train_count}'.encode(), label)
                        train_count += 1
                    else:
                        txn.put(f'validate:image:{validate_count}'.encode(), preprocessed_image.tobytes())
                        txn.put(f'validate:label:{validate_count}'.encode(), label)
                        validate_count += 1

            print(f"path {subdirectory_path} train {train_count} validate {validate_count} test {test_count}")

        txn.put('train_size'.encode(), f'{train_count}'.encode())
        txn.put('test_size'.encode(), f'{test_count}'.encode())
        txn.put('validate_size'.encode(), f'{validate_count}'.encode())

        print(train_count, test_count, validate_count, train_count + test_count + validate_count)
