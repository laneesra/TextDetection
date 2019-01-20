import csv


chain_fieldnames = ['image', 'isText', 'id', 'candidate count', 'average probability', 'average direction', 'size variation',
                   'distance variation', 'average axial ratio', 'average density', 'average width variation', 'colors']


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

