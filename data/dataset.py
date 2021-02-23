import cv2
import dlib
import numpy as np
import copy
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, Dataset


def make_preprocessing_dataset():
    path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/video"
    file_list = os.listdir(path)

    for i in file_list:
        cap = cv2.VideoCapture("C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/video/"+i)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/preprocessing_dataset/'+i, fourcc, 30.0, (64, 64))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("end or error")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            mod_frame = copy.copy(frame)
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                # cv2.rectangle(mod_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

                bPoint = []
                landmarks = predictor(frame, face)
                """
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(mod_frame, (x, y), 2, (0, 0, 255), -1)
                """

                for n in range(0, 17):
                    bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                for n in range(26, 18, -1):
                    bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                bpt = np.array(bPoint)

                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [bpt], (255, 255, 255))
                ROI = np.zeros_like(frame)
                ROI = cv2.copyTo(frame, mask, ROI)

                crop = ROI[y1:y2 + 10, x1 - 10:x2+10].copy()
                cropped = cv2.resize(crop, dsize=(64, 64))

            # cv2.imshow('frame', mod_frame)
            # cv2.imshow('ROI', ROI)
            cv2.imshow('cropped', cropped)
            out.write(cropped)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def make_maskset():
    path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/video"
    file_list = os.listdir(path)

    for i in file_list:
        cap = cv2.VideoCapture(path+'/'+i)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/maskset/'+i, fourcc, 30.0, (64, 64))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("end or error")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                bPoint = []
                landmarks = predictor(frame, face)

                for n in range(0, 17):
                    bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                for n in range(26, 18, -1):
                    bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                bpt = np.array(bPoint)

                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [bpt], (255, 255, 255))
                ROI = np.zeros_like(frame)
                ROI = cv2.bitwise_or(ROI, mask, ROI)

                crop = ROI[y1:y2 + 10, x1 - 10:x2+10].copy()
                cropped = cv2.resize(crop, dsize=(64, 64))

            cv2.imshow('cropped', cropped)
            out.write(cropped)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def make_dataset():
    path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/video"
    file_list = os.listdir(path)

    for i in file_list:
        cap = cv2.VideoCapture(path+'/'+i)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/dataset/'+i, fourcc, 30.0, (64, 64))

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("end or error")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                crop = frame[y1:y2 + 10, x1 - 10:x2+10].copy()
                cropped = cv2.resize(crop, dsize=(64, 64))

            cv2.imshow('cropped', cropped)
            out.write(cropped)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def make_image(option="data"):
    path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/video"
    file_list = os.listdir(path)

    for i in file_list:
        cap = cv2.VideoCapture(path+"/"+i)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        k = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("end or error")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            bPoint = []
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                if option == "data":
                    store_path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/data_image/"

                    crop = frame[y1:y2 + 10, x1 - 10:x2 + 10].copy()
                    cropped = cv2.resize(crop, dsize=(64, 64))
                elif option == "mask":
                    store_path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/mask_image/"

                    landmarks = predictor(frame, face)

                    for n in range(0, 17):
                        bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                    for n in range(26, 18, -1):
                        bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                    bpt = np.array(bPoint)

                    mask = np.zeros_like(frame)
                    cv2.fillPoly(mask, [bpt], (255, 255, 255))
                    ROI = np.zeros_like(frame)
                    ROI = cv2.bitwise_or(ROI, mask, ROI)

                    crop = ROI[y1:y2 + 10, x1 - 10:x2+10].copy()
                    cropped = cv2.resize(crop, dsize=(64, 64))
                elif option == "pre":
                    store_path = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/preprocessing_image/"
                    createFolder(store_path)

                    landmarks = predictor(frame, face)

                    for n in range(0, 17):
                        bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                    for n in range(26, 18, -1):
                        bPoint.append((landmarks.part(n).x, landmarks.part(n).y))
                    bpt = np.array(bPoint)

                    mask = np.zeros_like(frame)
                    cv2.fillPoly(mask, [bpt], (255, 255, 255))
                    ROI = np.zeros_like(frame)
                    ROI = cv2.copyTo(frame, mask, ROI)

                    crop = ROI[y1:y2 + 10, x1 - 10:x2 + 10].copy()
                    cropped = cv2.resize(crop, dsize=(64, 64))

            createFolder(store_path+i)
            cv2.imwrite(store_path+i+'/'+str(k)+".jpg", cropped)
            if cv2.waitKey(1) == ord('q'):
                break
            k += 1

        cap.release()
        cv2.destroyAllWindows()


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


class UBFCDataset(torch.utils.data.Dataset):
    def __init__(self, dir, data_path, mask_path, ppg_path, txt_file_name="/ground_truth.txt"):
        self.dir = dir
        self.data_path = data_path
        self.mask_path = mask_path
        self.ppg_path = ppg_path
        self.txt_file_name = txt_file_name

        self.data, self.data_file_name = get_image(self.dir, self.data_path)
        self.mask, self.mask_file_name = get_image(self.dir, self.mask_path)
        self.ppg = get_text(self.dir, self.ppg_path, self.txt_file_name)

        self.dic = {'image': self.data, 'mask': self.mask, 'ppg': self.ppg}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {'image': self.data[item], 'mask': self.mask[item], 'ppg': self.ppg[item]}


def imshow(img):
    img = img/2+0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

    print(np_img.shape)
    print((np.transpose(np_img, (1, 2, 0))).shape)


def get_image(dir, path):
    file_list = os.listdir(dir + path)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(root=dir + path, transform=trans)
    classes = trainset.classes

    label = []
    inputs = []
    for k in range(len(file_list)):
        inputs.append([])

    trainloader = DataLoader(trainset, batch_size=2852, shuffle=False, num_workers=5)
    for batch_idx, (input, targets) in enumerate(trainloader):
        input = input.tolist()
        for i in range(len(input)):  # input.size(0)
            label.append(classes[targets[i]])
            # label.append(targets[i])

            for k in range(len(file_list)):
                if targets[i] == k:
                    inputs[k].append(input[i])

    for k in range(len(file_list)):
        inputs[k] = torch.Tensor(inputs[k])

    return inputs, label


def get_text(dir, path, filename):
    file_list = os.listdir(dir + path)

    ppg = []
    seg = list()
    for i in file_list:
        # ppg.append(pd.read_csv(dir + path + i + filename, delim_whitespace=True, header=None))
        f = open(dir+path+i+filename, 'r')

        for word in f:
            seg = word.split('   ')

        seg = list(map(float, seg))

        seg = torch.Tensor(seg)
        ppg.append(seg)

        f.close()

    return ppg


if __name__ == "__main__":
    dir = "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/src/"
    data_path = "data_image"
    mask_path = "mask_image"
    ppg_path = "ppg/"
    txt_file_name = "/ground_truth.txt"

    dataset = UBFCDataset(dir, data_path, mask_path, ppg_path, txt_file_name)

    torch.save(dataset.dic, "C:/Users/intern/PycharmProjects/pythonProject14-metaRppg/data/UBFCDataset.pth")
