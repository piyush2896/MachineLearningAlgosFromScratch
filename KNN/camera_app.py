import numpy as np
import cv2
from knn import KNN

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = {}

def __get_data__():
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr

def collect_data(person_name, n=10):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1
        
        faces, fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w, :]
            if ix % skip_frame == 0:
                roi = cv2.resize(fc, (64, 64))
                data.append(roi)
            if len(data) >= n:
                flag = True
                break
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0))
        
        if flag or cv2.waitKey(1) == 27:
            break
        cv2.imshow('rgb', fr)
    
    data = np.asarray(data)
    print(data.shape)

    np.save(person_name+'.npy', data)
    cv2.destroyAllWindows()

def train_knn(person_name_list, k=2):
    person_to_labels = {}
    ix = 0
    for person in person_name_list:
        labels[ix] = person
        person_to_labels[person] = ix
        ix += 1
    
    labels_train = []
    data = []
    for person in person_name_list:
        curr_data = np.load(person + '.npy')
        labels_train.append([person_to_labels[person] for i in range(curr_data.shape[0])])
        data.append(curr_data.reshape((curr_data.shape[0], -1)))
        
    features_train = np.concatenate(data)
    labels_train = np.concatenate(labels_train)
    knn = KNN(k)
    knn.fit(features_train, labels_train)
    return knn
    

def predict(knn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1
        
        faces, fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w, :]
            
            roi = cv2.resize(fc, (64, 64))
            pred = knn.predict(np.array([roi.flatten()]))
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0))
            cv2.putText(fr, labels[int(pred)], (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('rgb', fr)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data('ABC')
    input('Press any key to collect 2nd data')
    collect_data('XYZ')
    input('Press any key to start Training...')
    knn = train_knn(['piyush','palak'])
    print('Training Complete')
    input('Press any key to start Predictions')
    predict(knn)
    cv2.destroyAllWindows()