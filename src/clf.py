from catboost import CatBoostClassifier, Pool
import Components_pb2 as pbcomp


def fit_catboost():
    train_file = '.././chains.df'
    cd_file = '.././chains.cd'
    train_pool = Pool(train_file, column_description=cd_file)
    model = CatBoostClassifier(depth=3, iterations=100, eval_metric='F1', task_type='CPU')
    model.fit(train_pool)
    model.save_model('chain.model')


def predict_catboost(is_dark_on_light):
    if is_dark_on_light:
        test_comp_file = '../comp/components_dark.df'
    else:
        test_comp_file = '../comp/components_light.df'

    cd_comp_file = '../components.cd'

    test_comp_pool = Pool(test_comp_file, column_description=cd_comp_file)

    comp_model = CatBoostClassifier().load_model('comp.model')
    preds = comp_model.predict(test_comp_pool)
    probas = comp_model.predict_proba(test_comp_pool)

    for i, proba in enumerate(probas):
        if preds[i]:
            print i, proba
    components = pbcomp.Components()
    write_preds_to_proto(components, preds, probas, is_dark_on_light)


def write_preds_to_proto(components, preds, probas, is_dark_on_light):
    if is_dark_on_light:
        f = open("../protobins/components_dark.bin", "rb")
    else:
        f = open("../protobins/components_light.bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    for i, comp in enumerate(components.components):
        comp.pred = preds[i]
        comp.proba = probas[i][1]
        if comp.pred:
            print(i, probas[i][1])

    if is_dark_on_light:
        f = open("../protobins/components_dark.bin", "wb")
    else:
        f = open("../protobins/components_light.bin", "wb")
    f.write(components.SerializeToString())
    f.close()


def predict_catboost_by_id(id, is_dark_on_light):
    if is_dark_on_light:
        test_comp_file = '../comp/components_dark_' + id + '.df'
    else:
        test_comp_file = '../comp/components_light_' + id + '.df'

    cd_comp_file = '../components.cd'
    try:
        test_comp_pool = Pool(test_comp_file, column_description=cd_comp_file)
    except BaseException:
        return

    comp_model = CatBoostClassifier().load_model('comp.model')
    preds = comp_model.predict(test_comp_pool)
    probas = comp_model.predict_proba(test_comp_pool)

    for i, proba in enumerate(probas):
        if preds[i]:
            print i, proba
    components = pbcomp.Components()
    write_preds_to_proto_by_id(id, components, preds, probas, is_dark_on_light)


def write_preds_to_proto_by_id(id, components, preds, probas, is_dark_on_light):
    if is_dark_on_light:
        f = open("../protobins/components_dark_" + id + ".bin", "rb")
    else:
        f = open("../protobins/components_light_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    for i, comp in enumerate(components.components):
        comp.pred = preds[i]
        comp.proba = probas[i][1]
        #if comp.pred:
         #   print(i, probas[i][1])

    if is_dark_on_light:
        f = open("../protobins/components_dark_" + id + ".bin", "wb")
    else:
        f = open("../protobins/components_light_" + id + ".bin", "wb")
    f.write(components.SerializeToString())
    f.close()

