import glob
import pickle

from catboost import CatBoostClassifier
from catboost import *
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_score

import Components_pb2 as pbcomp
import csv
import pandas
import numpy as np



comp_fieldnames = ['image', 'id', 'width', 'height', 'mean', 'standard deviation', 'width variation', 'aspect ratio',
                  'occupation ratio', 'minor axis', 'major axis', 'axial ratio', 'orientation', 'density',
                  'isDarkOnLight', 'text']
n = 81
k = 64

#train_id = ['0204', '0478', '1515', '1685', '1692', '1802', '2046', '2077', '1504', '1502', '1702'];
#train_id = ['2130', '2112', '1951', '1835', '1808', '1702', '1692', '1595', '1515', '1504', '1502', '0864', '0707', '0652', '0601', '0596']


def write_train_comp_to_csv():
    with open('components.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=comp_fieldnames)
        writer.writeheader()

        for id in train_id:
            comp_features = pandas.read_csv('./comp/component_IMG_' + id + '.csv').values
            for i in range(comp_features.shape[0]):
                row = {}
                for j in range(len(comp_fieldnames)):
                    row[comp_fieldnames[j]] = comp_features[i][j]
                writer.writerow(row)


def write_to_csv(id, components):
    with open('comp/component_IMG_' + id + '.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=comp_fieldnames)
        writer.writeheader()
        count = 0
        for comp in components.components:
            q = len(comp.points)
            S = comp.characteristic_scale
            if comp.minor_axis == 0 or comp.major_axis == 0:
                writer.writerow(
                    {'image': str(comp.image), 'id': comp.id, 'width': comp.width, 'height': comp.height,
                     'mean': round(comp.mean, 4), 'standard deviation': round(comp.SD, 4), 'width variation': round(comp.WV, 4),
                     'aspect ratio': round(comp.AR, 4), 'occupation ratio': round(comp.OR, 4), 'minor axis': round(comp.minor_axis, 4),
                     'major axis': round(comp.major_axis, 4), 'axial ratio': 0, 'orientation': round(comp.orientation, 4),
                     'density': 0, 'isDarkOnLight': comp.isDarkOnLight, 'text': comp.isText})
            else:
                writer.writerow(
                    {'image': str(comp.image), 'id': comp.id, 'width': comp.width, 'height': comp.height,
                     'mean': round(comp.mean, 4), 'standard deviation': round(comp.SD, 4), 'width variation': round(comp.WV, 4),
                     'aspect ratio': round(comp.AR, 4), 'occupation ratio': round(comp.OR, 4), 'minor axis': round(comp.minor_axis, 4),
                     'major axis': round(comp.major_axis), 'axial ratio': round(comp.minor_axis / comp.major_axis, 4),
                     'orientation': round(comp.orientation, 4), 'density': round(float(q)/S**2, 4),
                     'isDarkOnLight': comp.isDarkOnLight, 'text': comp.isText})
            count += 1


def preprocess(img):
    SZ = 20
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img


def write_train_hog_to_csv():
    fieldnames = ['hog' + str(i) for i in range(0, n)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_hog.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for id in train_id:
            print(id)
            hog_features = pandas.read_csv('./hog/component_hog_IMG_' + id + '.csv').values
            for i in range(hog_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, n):
                    row[fieldnames[j + 2]] = hog_features[i][j + 2]
                writer.writerow(row)


def extract_hog(path_dir):
    hog = cv.HOGDescriptor((20, 20), (10, 10), (5, 5), (10, 10), 9, 1, -1, 0, 0.2, 1, 64, True)
    list_hog = []
    for i in range(0, np.size(path_dir)):
        img = cv.imread(path_dir[i], 0)
        imresize = preprocess(img)
        ihog = hog.compute(imresize)
        list_hog.append(ihog)
    hog_features = np.squeeze(list_hog)
    return hog_features


def write_hog_to_csv(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    hog = extract_hog(comp_dir)
    fieldnames = ['hog' + str(i) for i in range(0, n)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('hog/component_hog_IMG_' + id + '.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(hog)):
            row = {'image': id, 'id': i}
            for j in range(0, n):
                row[fieldnames[j + 2]] = hog[i][j]
            writer.writerow(row)

k = 64
def extract_train_surf_to_csv():
    surf = cv.xfeatures2d.SURF_create(300)
    img_descs = []
    descriptors = []
    count = [0]
    for i, id in enumerate(train_id):
        print id + 'desc'
        path_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'),
                          key=get_num)
        for i in range(0, np.size(path_dir)):
            img = cv.imread(path_dir[i])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, desc = surf.detectAndCompute(gray, None)
            if desc is not None:
                img_descs.append((path_dir[i], desc))
                for d in desc:
                    descriptors.append(d)
            else:
                img_descs.append((path_dir[i], []))
        count.append(len(img_descs))

    filename = 'surf_model.model'
    '''kmeans = KMeans(n_clusters=k, random_state=241)
    kmeans.fit(descriptors)
    pickle.dump(kmeans, open(filename, 'wb'))'''

    kmeans = pickle.load(open(filename, 'rb'))
    im_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        print img_descs[i][0]
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                im_features[i][w] += 1

    fieldnames = ['surf' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    for l, id in enumerate(train_id):
        with open('surf/component_surf_IMG_' + id + '.csv', 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]
                writer.writerow(row)


def extract_train_sift_to_csv():
    sift = cv.xfeatures2d.SIFT_create()
    img_descs = []
    descriptors = []
    count = [0]
    for i, id in enumerate(train_id):
        print id + 'desc'
        path_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'),
                          key=get_num)
        for i in range(0, np.size(path_dir)):
            img = cv.imread(path_dir[i])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            if desc is not None:
                img_descs.append((path_dir[i], desc))
                for d in desc:
                    descriptors.append(d)
            else:
                img_descs.append((path_dir[i], []))
        count.append(len(img_descs))


    filename = 'sift_model.model'
  #  kmeans = KMeans(n_clusters=k, random_state=241)
  #  kmeans.fit(descriptors)
  #  pickle.dump(kmeans, open(filename, 'wb'))

    kmeans = pickle.load(open(filename, 'rb'))
    im_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        print img_descs[i][0]
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                im_features[i][w] += 1

    fieldnames = ['sift' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    for l, id in enumerate(train_id):
        with open('sift/component_sift_IMG_' + id + '.csv', 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]
                writer.writerow(row)


def extract_train_orb_to_csv():
    orb = cv.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                        scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
    img_descs = []
    descriptors = []
    count = [0]
    for id in train_id:
        print id + 'desc'
        path_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'),
                          key=get_num)

        for i in range(np.size(path_dir)):
            img = cv.imread(path_dir[i])
            kp, desc = orb.detectAndCompute(img, None)
            if desc is not None:
                img_descs.append((path_dir[i], desc))
                for d in desc:
                    descriptors.append(d)
            else:
                img_descs.append((path_dir[i], []))
        count.append(len(img_descs))

    filename = 'orb_model.model'
   # kmeans = KMeans(n_clusters=k, random_state=241)
  #  kmeans.fit(descriptors)
 #   pickle.dumpkmeans, open(filename, 'wb'))

    kmeans = pickle.load(open(filename, 'rb'))
    im_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        print img_descs[i][0]
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                im_features[i][w] += 1

    fieldnames = ['orb' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    for l, id in enumerate(train_id):
        with open('orb/component_orb_IMG_' + id + '.csv', 'w') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            writer.writeheader()

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]
                writer.writerow(row)



def write_train_surf_to_csv():
    fieldnames = ['surf' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_surf.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for id in train_id:
            print(id)
            sift_features = pandas.read_csv('./surf/component_surf_IMG_' + id + '.csv').values
            for i in range(sift_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = sift_features[i][j + 2]
                writer.writerow(row)


def write_surf_to_csv(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    surf = extract_surf(comp_dir)
    fieldnames = ['surf' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('surf/component_surf_IMG_' + id + '.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(surf)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j + 2]] = surf[i][j]
            writer.writerow(row)


def extract_surf(path_dir):
    surf = cv.xfeatures2d.SURF_create(300)
    img_descs = []
    descriptors = []

    for i in range(np.size(path_dir)):
        img = cv.imread(path_dir[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, desc = surf.detectAndCompute(gray, None)
        if desc is not None:
            img_descs.append((path_dir[i], desc))
            for d in desc:
                descriptors.append(d)
        else:
            img_descs.append((path_dir[i], []))

    filename = 'surf_model.model'
    kmeans = pickle.load(open(filename, 'rb'))
    img_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                img_features[i][w] += 1

    return img_features


def write_train_sift_to_csv():
    fieldnames = ['sift' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_sift.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for id in train_id:
            sift_features = pandas.read_csv('./sift/component_sift_IMG_' + id + '.csv').values
            for i in range(sift_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = sift_features[i][j + 2]
                writer.writerow(row)



def write_sift_to_csv(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    sift = extract_sift(comp_dir)
    fieldnames = ['sift' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('sift/component_sift_IMG_' + id + '.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(sift)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j+2]] = sift[i][j]
            writer.writerow(row)


def extract_sift(path_dir):
    sift = cv.xfeatures2d.SIFT_create()
    img_descs = []
    descriptors = []

    for i in range(np.size(path_dir)):
        img = cv.imread(path_dir[i])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        if desc is not None:
            img_descs.append((path_dir[i], desc))
            for d in desc:
                descriptors.append(d)
        else:
            img_descs.append((path_dir[i], []))

    filename = 'sift_model.model'
    kmeans = pickle.load(open(filename, 'rb'))
    img_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                img_features[i][w] += 1

    return img_features



def write_train_orb_to_csv():
    fieldnames = ['orb' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_orb.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for id in train_id:
            orb_features = pandas.read_csv('./orb/component_orb_IMG_' + id + '.csv').values
            for i in range(orb_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = orb_features[i][j + 2]
                writer.writerow(row)


def write_orb_to_csv(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    orb = extract_orb(comp_dir)
    fieldnames = ['orb' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('orb/component_orb_IMG_' + id + '.csv', 'w') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(orb)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j+2]] = orb[i][j]
            writer.writerow(row)


def extract_orb(path_dir):
    orb = cv.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,
                        scoreType=cv.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
    img_descs = []
    descriptors = []

    for i in range(np.size(path_dir)):
        img = cv.imread(path_dir[i])
        kp, desc = orb.detectAndCompute(img, None)
        if desc is not None:
            img_descs.append((path_dir[i], desc))
            for d in desc:
                descriptors.append(d)
        else:
            img_descs.append((path_dir[i], []))

    filename = 'orb_model.model'
    kmeans = pickle.load(open(filename, 'rb'))
    img_features = np.zeros((len(img_descs), k), "int32")

    for i in xrange(len(img_descs)):
        if len(img_descs[i][1]) != 0:
            words = kmeans.predict(img_descs[i][1])
            for w in words:
                img_features[i][w] += 1

    return img_features


def get_num(path):
    return int(path[68:-4])


def get_num_x(path):
    return int(path[75:-4])


def train_catboost(train_features, target):
    best_recall = 0
    best_f1 = 0
    best_n = 0
    kf = KFold(n_splits=10, shuffle=False, random_state=1)

    for n in np.linspace(3, 300, 10):
        model = CatBoostClassifier(iterations=int(n), depth=3, learning_rate=0.5, loss_function='Logloss', task_type='GPU')
        #model.fit(train_features, target, cat_features=[0])
        recall_scores = cross_val_score(model, train_features, target, cv=kf, scoring='recall')
        recall = recall_scores.mean()
        print recall
        #preds_class = model.predict(test_features)
        #recall = recall_score(target, preds_class)
        #precision = precision_score(target, preds_class)
        #f1 = f1_score(target, preds_class)
        if (best_recall < recall):
            best_recall = recall
            best_n = n


    print (best_f1, best_recall, best_n)
    return best_f1, best_recall, best_n


def predict_catboost(test_features, filename):
    model = CatBoostClassifier().load_model(filename)
    preds = model.predict(test_features)
    probas = model.predict_proba(test_features)

    return preds, probas


def fit_catboost(train_features, target, n, depth, lr, task, filename):
    model = CatBoostClassifier(iterations=int(n), depth=depth, learning_rate=lr, loss_function='Logloss', task_type=task)
    model.fit(train_features, target, cat_features=[0])
    model.save_model(filename)


def write_preds_to_proto(components, preds):
    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    count = 0
    for comp in components.components:
        comp.pred = preds[count]
        count += 1

    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "wb")
    f.write(components.SerializeToString())
    f.close()


def write_train_to_csv():
    write_train_sift_to_csv()
    #write_train_hog_to_csv()
    #write_train_orb_to_csv()
    write_train_surf_to_csv()


def write_test_to_csv(id, components):
    write_to_csv(id, components)
    write_sift_to_csv(id)
    write_hog_to_csv(id)
    write_orb_to_csv(id)
    write_surf_to_csv(id)

dirs = sorted(glob.glob('/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/catboost/IMG_*.JPG'), key=get_num_x)
train_id = [dir[75:-4] for dir in dirs]


#extract_train_surf_to_csv()
#extract_train_sift_to_csv()
#extract_train_orb_to_csv()
#write_train_to_csv()

'''train_sift_features = pandas.read_csv('./components_sift.csv', index_col='id')
train_hog_features = pandas.read_csv('./components_hog.csv', index_col='id')
train_comp_features = pandas.read_csv('./components.csv', index_col='id')
train_surf_features = pandas.read_csv('./components_surf.csv', index_col='id')
train_orb_features = pandas.read_csv('./components_orb.csv', index_col='id')

train_target = train_comp_features['text'].values
train_comp_features.drop('text', axis=1, inplace=True)

fit_catboost(train_comp_features, train_target, n=36, depth=2, lr=1, task='CPU', filename='comp_catboost.model')
fit_catboost(train_sift_features, train_target, n=135, depth=3, lr=0.5, task='GPU', filename='sift_catboost.model')
fit_catboost(train_surf_features, train_target, n=3, depth=2, lr=1, task='CPU', filename='surf_catboost.model')
fit_catboost(train_orb_features, train_target, n=3, depth=2, lr=1, task='CPU', filename='orb_catboost.model')
fit_catboost(train_hog_features, train_target, n=234, depth=3, lr=0.5, task='GPU', filename='hog_catboost.model')'''

#train_catboost(train_comp_features, train_target)
#train_catboost(train_hog_features, train_target)
#train_catboost(train_comp_features, train_target)
#train_catboost(train_surf_features, train_target)
#train_catboost(train_orb_features, train_target)

id = '2220' #2075
#train_dir = glob.glob('/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/test/IMG_' + id + '.JPG')
components = pbcomp.Components()
f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
components.ParseFromString(f.read())
f.close()
write_test_to_csv(id, components)

test_sift_features = pandas.read_csv('./sift/component_sift_IMG_' + id + '.csv', index_col='id')
test_orb_features = pandas.read_csv('./orb//component_orb_IMG_' + id + '.csv', index_col='id')
test_surf_features = pandas.read_csv('./surf/component_surf_IMG_' + id + '.csv', index_col='id')
test_hog_features = pandas.read_csv('./hog/component_hog_IMG_' + id + '.csv', index_col='id')
test_comp_features = pandas.read_csv('./comp/component_IMG_' + id + '.csv', index_col='id')
test_target = test_comp_features['text'].values
test_comp_features.drop('text', axis=1, inplace=True)

sift_pred, sift_proba = predict_catboost(test_sift_features, 'sift_catboost.model')
orb_pred, orb_proba = predict_catboost(test_orb_features, 'orb_catboost.model')
surf_pred, surf_proba = predict_catboost(test_surf_features, 'surf_catboost.model')
hog_pred, hog_proba = predict_catboost(test_hog_features, 'hog_catboost.model')
comp_pred, comp_proba = predict_catboost(test_comp_features, 'comp_catboost.model')
zip_pred = [(sift_pred[i], hog_pred[i], comp_pred[i], surf_pred[i], orb_pred[i]) for i in range(len(sift_pred))]

proba = []
preds = []
for i in zip_pred:
    print np.mean(i)
    if np.mean(i) >= 0.5:
        preds.append(1)
    else:
        preds.append(0)
print preds

for i in test_target:
    if i:
        i = 1
    else:
        i = 0

print test_target
print sift_pred
print orb_pred
print surf_pred
print hog_pred
print comp_pred
#preds = comp_pred
print(f1_score(test_target, preds))
print(recall_score(test_target, preds))
print(precision_score(test_target, preds))
write_preds_to_proto(components, np.array(preds, dtype=int))

#print(surf_prec, surf_f1, surf_score, surf_n)
#print(orb_prec, orb_f1, orb_score, orb_n)
#print(sift_prec, sift_f1, sift_score, sift_n)
#print(hog_prec, hog_f1, hog_score, hog_n)
#print(comp_prec, comp_f1, comp_score, comp_n)