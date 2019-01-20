import csv
import glob
import pandas


def get_num(path):
    return int(path[66:-4])


chain_fieldnames = ['image', 'isText', 'id', 'candidate count', 'average probability', 'average direction', 'size variation',
                   'distance variation', 'average axial ratio', 'average density', 'average width variation', 'colors']
dirs = sorted(glob.glob('/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/IMG_*.JPG'), key=get_num)
train_id = [dir[66:-4] for dir in dirs]


def write_train_chains_to_df():
    with open('../chains.df', 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=chain_fieldnames)
        writer.writeheader()

        for id in train_id:
            try:
                chain_features = pandas.read_csv('../chain/chain_dark_' + id + '.df').values
                print id, chain_features.shape[0]
                for i in range(chain_features.shape[0]):
                    row = chain_features[i]
                    f.write('\n'.join('\n'+row))
            except BaseException:
                continue

            try:
                chain_features = pandas.read_csv('../chain/chain_light_' + id + '.df').values
                print id, chain_features.shape[0]

                for i in range(chain_features.shape[0]):
                    row = chain_features[i]
                    f.write('\n'.join('\n'+row))
            except BaseException:
                continue


def write_chain_to_df(candidate_count, average_probability, size_variation, distance_variation, average_axial_ratio,
                      average_density, average_width_variation, colors, is_dark_on_light):
    if is_dark_on_light:
        filename = 'chain_dark.df'
    else:
        filename = 'chain_light.df'

    with open('../chain/' + filename, 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=chain_fieldnames)

        for i in range(len(candidate_count)):
            writer.writerow(
                {'image': 0, 'isText': 0, 'id': i, 'candidate count': candidate_count[i], 'average probability': average_probability[i],
                 'size variation': size_variation[i], 'distance variation': distance_variation[i],
                 'average axial ratio': average_axial_ratio[i], 'average density': average_density[i],
                 'average width variation': average_width_variation[i], 'colors': colors[i]})


def write_train_chain_to_df(id, candidate_count, average_probability, size_variation, distance_variation,
                            average_axial_ratio, average_density, average_width_variation, colors, is_dark_on_light):
    if is_dark_on_light:
        filename = 'chain_dark_' + id + '.df'
    else:
        filename = 'chain_light_' + id + '.df'
    with open('../chain/' + filename, 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=chain_fieldnames)
        writer.writeheader()

        for i in range(len(candidate_count)):
            writer.writerow(
                {'image': id, 'isText': 0, 'id': i, 'candidate count': candidate_count[i], 'average probability': average_probability[i],
                 'size variation': size_variation[i], 'distance variation': distance_variation[i],
                 'average axial ratio': average_axial_ratio[i], 'average density': average_density[i],
                 'average width variation': average_width_variation[i], 'colors': colors[i]})


def get_num_x(path):
    return int(path[75:-4])

