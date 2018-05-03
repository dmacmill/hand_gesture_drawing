#!/usr/bin/python

import csv
import cv2
import glob
import os
from shutil import copy

data_dir = "./dataset/"


def get_subjects():
    subjects_dirs = data_dir + 'labels/*'
    subjects = []
    for subject in glob.glob(subjects_dirs):
        subjects.append(subject)
    return subjects

def get_scenes(subject):
    scenes_dirs = data_dir + 'labels/' + subject + '/*'
    scenes = []
    for scene in glob.glob(subject + '/*'):
        scenes.append(scene)

    return scenes

def get_groups(subject, scene):
    groups = []
    for group in glob.glob(scene + '/*'):
        components = group.split('/')
        groupname = components[-1]
        scenename = components[-2]
        subjectname = components[-3]
#        print subjectname
#        print scenename
#        print groupname

        groups.append(list([subjectname, scenename, groupname]))

    return groups

def get_all_data_organized():
    all_data = []
    subjects = get_subjects()

    for subject in subjects:
        scenes = get_scenes(subject)
        for scene in scenes:
            all_data += get_groups(subject, scene)
    
    return all_data



def get_all_labels_raw():
    all_data = []

    all_files_glob = data_dir + '*/*/*.csv'

    all_files = glob.glob(all_files_glob)
    for f in all_files:
        print f
        with open(f, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                row = list(row)
                row.append('yolo')
                all_data += row
                if row[0] == '49':
                    print row

    return all_data

def get_all_labels_organized(groups):
    labels = []
    labeldir = data_dir + 'labels/'

    for g in groups:
        f = labeldir + g[0] + '/' + g[1] + '/' + g[2]
        with open(f, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                row = list(row)
                row += g
                labels.append(row)

    return labels

def get_frames_for_gesture(gesture, labels):
    frames = []
    count = 0
    all_data = list(filter(lambda x: x[0] == gesture, labels))

    processed_dir = data_dir + 'processed/' + gesture + '/'

    if not os.path.exists(processed_dir):
        print "make", processed_dir
        os.makedirs(processed_dir)

    for section in all_data:
        vid_num = filter(str.isdigit, section[5])
        video_path = data_dir + 'videos/' + section[3].title() + '/' + section[4] + '/Color/rgb' + vid_num + '.avi'

        cap = cv2.VideoCapture(video_path)
        begin = int(section[1])
        end = int(section[2])
        length = end - begin 
        buffer = length/3

        for i in range(begin, begin + buffer): #, end - buffer):
            count += 1
            cap.set(1, i)
            _, frame = cap.read()
            frames.append(frame)
            cv2.imwrite(processed_dir + str(count).zfill(5) + '.png', frame)            
    return count

def get_images_for_gesture(gesture, labels):
    frames = []
    count = 0
    all_data = list(filter(lambda x: x[0] == gesture, labels))

    processed_dir = data_dir + 'processed/' + gesture + '/'

    if not os.path.exists(processed_dir):
        print "make", processed_dir
        os.makedirs(processed_dir)

    for section in all_data:
        grp_num = filter(str.isdigit, section[5])
        image_dir = data_dir + 'images/' + section[3].title() + '/' + section[4] + '/Color/rgb' + grp_num + '/'
        begin = int(section[1])
        end = int(section[2])
        length = end - begin 
        buffer = length/3
	half = length/2
#       for i in range(begin + buffer, end - buffer):
        copy(image_dir+str(begin+half).zfill(6)+'.jpg', processed_dir + str(count).zfill(6) + '.jpg')
        count += 1

    return count


groups = get_all_data_organized()
labels = get_all_labels_organized(groups)


# all_labels = get_all_labels_raw()
frames = get_frames_for_gesture('34', labels)
# frames = get_images_for_gesture('X', labels)

for i in range(21, 31):
    frames = get_images_for_gesture(str(i), labels)
    print i, ": ", frames

