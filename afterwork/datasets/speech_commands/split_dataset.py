import os
import shutil
import glob
from distutils.dir_util import copy_tree, mkpath

train_valid_ratio = 0.8

root_dir = "./speech_commands_v2"

subdirs = next(os.walk(root_dir))[1]
subdirs.remove("_background_noise_")

all_classes = subdirs

train_classes = ['right', 'eight', 'cat', 'tree', 'backward',
               'learn', 'bed', 'happy', 'go', 'dog', 
               'no', 'wow', 'follow', 'nine', 'left',
               'on', 'five', 'forward', 'off', 'four']

test_classes = [label for label in all_classes if label not in train_classes]

print(train_classes)
print(test_classes)

train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'

mkpath(train_dir)
mkpath(validation_dir)
mkpath(test_dir)

for label in train_classes:
    filenames = os.listdir(os.path.join(root_dir, label))
    filenames.sort()

    count_by_actor = {}
    count_entries = 0

    for filename in filenames:
        actor_id = filename.split('_')[0]
        count_entries += 1

        if actor_id in count_by_actor.keys():
            count_by_actor[actor_id] += 1
        else:
            count_by_actor[actor_id] = 1

    train_words_number = count_entries * train_valid_ratio
    count_train_words = 0


    mkpath(os.path.join(train_dir, label))
    mkpath(os.path.join(validation_dir, label))

    for actor_id in count_by_actor.keys():
        dest_path = ""

        if count_train_words < train_words_number:
            dest_path = os.path.join(train_dir, label)
            count_train_words += count_by_actor[actor_id]
        else:
            dest_path = os.path.join(validation_dir, label)

        for file in glob.glob(os.path.join(root_dir, label, actor_id + '*')):
            shutil.copy(file, dest_path)    

for label in test_classes:
    copy_tree(os.path.join(root_dir, label), os.path.join(test_dir, label))
