from catboost import CatBoostClassifier, Pool
import Components_pb2 as pbcomp

'''fitting model and saving it'''
def fit_chain():
    train_file = '.././chains.df'
    cd_file = '.././chains.cd'
    train_pool = Pool(train_file, column_description=cd_file)
    model = CatBoostClassifier(depth=3, iterations=100, eval_metric='F1', task_type='CPU')
    model.fit(train_pool)
    model.save_model('chain.model')

'''predicting class of components'''
def predict_comp(is_dark_on_light):
    if is_dark_on_light:
        test_comp_file = '../comp/components_dark.df'
    else:
        test_comp_file = '../comp/components_light.df'

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
    write_preds_to_proto(components, preds, probas, is_dark_on_light)

'''predicting class of chains'''
def predict_chain(is_dark_on_light):
    if is_dark_on_light:
        test_comp_file = '../chain/chain_dark.df'
    else:
        test_comp_file = '../chain/chain_light.df'

    cd_chain_file = '../chains.cd'
    try:
        test_chain_pool = Pool(test_comp_file, column_description=cd_chain_file)
    except BaseException:
        return

    comp_model = CatBoostClassifier().load_model('chain.model')
    preds = comp_model.predict(test_chain_pool)

    return preds

'''writing predictions to proto bin file'''
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

