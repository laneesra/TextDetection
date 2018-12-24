import glob

import Components_pb2 as pbcomp
import csv
import pandas


def write_train_comp_to_df():
    with open('../components.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=comp_fieldnames)
        writer.writeheader()

        for id in train_id:
            comp_features = pandas.read_csv('../comp/component_IMG_' + id + '.csv').values
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


def get_num(path):
    return int(path[68:-4])


def get_num_x(path):
    return int(path[75:-4])


def write_train_to_df():
    for id in train_id:
        components = pbcomp.Components()
        f = open("../protobins/component_IMG_" + id + ".bin", "rb")
        components.ParseFromString(f.read())
        f.close()
        write_comp_to_df(id, components)
    write_train_comp_to_df()


def write_test_to_df(id, components):
    write_comp_to_df(id, components)


def run(id):
    f = open("../protobins/component_IMG_" + id + ".bin", "rb")
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
