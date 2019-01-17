import glob

import feature_extraction as fe
import clf
import text_localizing as tl
import chain_analysis as ca


def execute(is_dark_on_light):
    fe.run(is_dark_on_light)
    clf.predict_catboost(is_dark_on_light)
    components = tl.read_component(is_dark_on_light)
    tl.text_localizing(components, is_dark_on_light)
    # todo add filtering


def execute_by_id(id, is_dark_on_light):
    fe.train_run(id, is_dark_on_light)
    clf.predict_catboost_by_id(id, is_dark_on_light)
    components = tl.read_component_by_id(id, is_dark_on_light)
    if len(components.components):
        candidate_count, average_probability, size_variation, distance_variation, average_axial_ratio, \
            average_density, average_width_variation, colors = tl.train_text_text_localizing(components, is_dark_on_light)
        ca.write_train_chain_to_df(id, candidate_count, average_probability, size_variation, distance_variation,
                                   average_axial_ratio, average_density, average_width_variation, colors, is_dark_on_light)


def make_chain_dataset():
    dirs = sorted(glob.glob('/home/laneesra/PycharmProjects/TextDetection/MSRA-TD500/train/IMG_*.JPG'),
                  key=ca.get_num)
    train_id = [dir[66:-4] for dir in dirs]
    for id in train_id:
        if id > '1600':
            print(id)
            execute_by_id(id, True)
            execute_by_id(id, False)

    ca.write_train_chains_to_df()


#make_chain_dataset()
clf.fit_catboost()

#execute(True)
#execute(False)
