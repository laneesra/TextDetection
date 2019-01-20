import feature_extraction as fe
import clf
import text_localizing as tl
import write_chain_features as ca


def execute(is_dark_on_light):
    fe.run(is_dark_on_light)
    clf.predict_comp(is_dark_on_light)
    components = tl.read_component(is_dark_on_light)
    filename = components.components[0].filename
    if len(components.components):
        candidate_count, average_probability, size_variation, distance_variation, average_axial_ratio, \
            average_density, average_width_variation, colors, lines = tl.text_localizing(components, is_dark_on_light)
        ca.write_chain_to_df(candidate_count, average_probability, size_variation, distance_variation,
                             average_axial_ratio, average_density, average_width_variation, colors, is_dark_on_light)
        preds = clf.predict_chain(is_dark_on_light)
        tl.final_text(filename, preds, lines, is_dark_on_light)


execute(True)
execute(False)
