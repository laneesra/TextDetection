import glob
import pickle

import Components_pb2 as pbcomp
import csv
import pandas
import numpy as np
import cv2 as cv


def write_train_comp_to_df():
    with open('components.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=comp_fieldnames)
        writer.writeheader()

        for id in train_id:
            comp_features = pandas.read_csv('./comp/component_IMG_' + id + '.csv').values
            for i in range(comp_features.shape[0]):
                row = {}
                for j in range(len(comp_fieldnames)):
                    row[comp_fieldnames[j]] = comp_features[i][j]
                writer.writerow(row)


def write_comp_to_df(id, components):
    with open('../comp/component_IMG_' + id + '.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=comp_fieldnames)
        count = 0
        for comp in components.components:
            q = len(comp.points)
            S = comp.minor_axis + comp.major_axis
            if comp.minor_axis == 0 or comp.major_axis == 0:
                writer.writerow(
                    {'image': str(comp.image), 'id': comp.id, 'width': comp.width, 'height': comp.height,
                     'mean': round(comp.mean, 4), 'standard deviation': round(comp.SD, 4), 'width variation': round(comp.WV, 4),
                     'aspect ratio': round(comp.AR, 4), 'occupation ratio': round(comp.OR, 4), 'minor axis': round(comp.minor_axis, 4),
                     'major axis': round(comp.major_axis, 4), 'axial ratio': 0, 'orientation': round(comp.orientation, 4),
                     'density': 0, 'isDarkOnLight': int(comp.isDarkOnLight), 'text': int(comp.isText)})
            else:
                writer.writerow(
                    {'image': str(comp.image), 'id': comp.id, 'width': comp.width, 'height': comp.height,
                     'mean': round(comp.mean, 4), 'standard deviation': round(comp.SD, 4), 'width variation': round(comp.WV, 4),
                     'aspect ratio': round(comp.AR, 4), 'occupation ratio': round(comp.OR, 4), 'minor axis': round(comp.minor_axis, 4),
                     'major axis': round(comp.major_axis), 'axial ratio': round(comp.minor_axis / comp.major_axis, 4),
                     'orientation': round(comp.orientation, 4), 'density': round(float(q)/S**2, 4),
                     'isDarkOnLight': int(comp.isDarkOnLight), 'text': int(comp.isText)})
            count += 1


def extract_train_surf_to_df():
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
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    for l, id in enumerate(train_id):
        with open('surf/component_surf_IMG_' + id + '.df', 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]
                row['text'] = 0
                writer.writerow(row)


def extract_train_sift_to_df():
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
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    for l, id in enumerate(train_id):
        with open('sift/component_sift_IMG_' + id + '.df', 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]
                row['text'] = 0

                writer.writerow(row)


def extract_train_orb_to_df():
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
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    for l, id in enumerate(train_id):
        with open('orb/component_orb_IMG_' + id + '.df', 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

            for i in range(count[l], count[l + 1]):
                row = {'image': id, 'id': i}
                for j in range(0, k):
                    row[fieldnames[j + 2]] = im_features[i][j]

                row['text'] = 0
                writer.writerow(row)



def write_train_surf_to_df():
    fieldnames = ['surf' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_surf.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for id in train_id:
            print(id)
            sift_features = pandas.read_csv('./surf/component_surf_IMG_' + id + '.df').values
            for i in range(sift_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = sift_features[i][j + 2]
                writer.writerow(row)


def write_surf_to_df(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    surf = extract_surf(comp_dir)
    fieldnames = ['surf' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    with open('surf/component_surf_IMG_' + id + '.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for i in range(0, len(surf)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j + 2]] = surf[i][j]
            row['text'] = 0
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


def write_train_sift_to_df():
    fieldnames = ['sift' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames

    with open('components_sift.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for id in train_id:
            sift_features = pandas.read_csv('./sift/component_sift_IMG_' + id + '.df').values
            for i in range(sift_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = sift_features[i][j + 2]
                writer.writerow(row)


def write_sift_to_df(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    sift = extract_sift(comp_dir)
    fieldnames = ['sift' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    with open('sift/component_sift_IMG_' + id + '.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for i in range(0, len(sift)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j+2]] = sift[i][j]
            row['text'] = 0
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


def write_train_orb_to_df():
    fieldnames = ['orb' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    with open('components_orb.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for id in train_id:
            orb_features = pandas.read_csv('./orb/component_orb_IMG_' + id + '.df').values
            for i in range(orb_features.shape[0]):
                row = {'image': id, 'id': i}

                for j in range(0, k):
                    row[fieldnames[j + 2]] = orb_features[i][j + 2]
                row['text'] = 0
                writer.writerow(row)


def write_orb_to_df(id):
    comp_dir = sorted(glob.glob('/home/laneesra/CLionProjects/TextDetection/components/IMG_' + id + '/COMP_*.JPG'), key=get_num)
    print(comp_dir)
    orb = extract_orb(comp_dir)
    fieldnames = ['orb' + str(i) for i in range(0, k)]
    fieldnames = ['image', 'id'] + fieldnames + ['text']

    with open('orb/component_orb_IMG_' + id + '.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)

        for i in range(0, len(orb)):
            row = {'image': id, 'id': i}
            for j in range(0, k):
                row[fieldnames[j+2]] = orb[i][j]
            row['text'] = 0
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


def write_train_to_df():
    extract_train_sift_to_df()
    extract_train_orb_to_df()
    extract_train_surf_to_df()

    for id in train_id:
        components = pbcomp.Components()
        f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
        components.ParseFromString(f.read())
        f.close()
        write_comp_to_df(id, components)

    write_train_sift_to_df()
    write_train_orb_to_df()
    write_train_surf_to_df()
    write_train_comp_to_df()


def write_test_to_df(id, components):
    write_comp_to_df(id, components)
  #  write_sift_to_df(id)
  # write_orb_to_df(id)
  #  write_surf_to_df(id)


def run(id):
    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()
    write_test_to_df(id, components)


comp_fieldnames = ['image', 'id', 'width', 'height', 'mean', 'standard deviation', 'width variation', 'aspect ratio',
                  'occupation ratio', 'minor axis', 'major axis', 'axial ratio', 'orientation', 'density',
                  'isDarkOnLight', 'text']
dirs = sorted(glob.glob('/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/catboost/IMG_*.JPG'), key=get_num_x)
train_id = [dir[75:-4] for dir in dirs]
k = 64
components = pbcomp.Components()
run('0495')