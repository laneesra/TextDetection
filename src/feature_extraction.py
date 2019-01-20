import glob

import Components_pb2 as pbcomp
import csv


def write_comp_to_df(components, is_dark_on_light):
    if is_dark_on_light:
        filename = '../comp/components_dark.df'
    else:
        filename = '../comp/components_light.df'
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=comp_fieldnames)

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


def run(is_dark_on_light):
    if is_dark_on_light:
        f = open("../protobins/components_dark.bin", "rb")
    else:
        f = open("../protobins/components_light.bin", "rb")
    components.ParseFromString(f.read())
    f.close()
    write_comp_to_df(components, is_dark_on_light)


comp_fieldnames = ['image', 'id', 'width', 'height', 'mean', 'standard deviation', 'width variation', 'aspect ratio',
                  'occupation ratio', 'minor axis', 'major axis', 'axial ratio', 'orientation', 'density',
                  'isDarkOnLight', 'text']
   components = pbcomp.Components()
