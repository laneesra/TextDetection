from catboost import CatBoostClassifier, Pool
import Components_pb2 as pbcomp


def predict_catboost():
    test_comp_file = '../comp/component_IMG.df'
    cd_comp_file = '../components.cd'

    test_comp_pool = Pool(test_comp_file, column_description=cd_comp_file)

    comp_model = CatBoostClassifier().load_model('comp.model')
    preds = comp_model.predict(test_comp_pool)
    probas = comp_model.predict_proba(test_comp_pool)

    for i, proba in enumerate(probas):
        if preds[i]:
            print i, proba
    components = pbcomp.Components()
    write_preds_to_proto(components, preds)


def write_preds_to_proto(components, preds):
    f = open("../protobins/components.bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    for i, comp in enumerate(components.components):
        comp.pred = preds[i]
        if comp.pred:
            print(i)

    f = open("../protobins/components.bin", "wb")
    f.write(components.SerializeToString())
    f.close()
