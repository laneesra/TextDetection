from catboost import CatBoostClassifier, Pool
import numpy as np
import Components_pb2 as pbcomp


def fit_catboost():
    train_comp_file = '.././components.df'
    train_surf_file = '.././components_surf.df'
    train_sift_file = '.././components_sift.df'
    train_orb_file = '.././components_orb.df'

    cd_comp_file = '.././components.cd'
    cd_other_file = '.././sift.cd'

    train_comp_pool = Pool(train_comp_file, column_description=cd_comp_file)
    train_surf_pool = Pool(train_surf_file, column_description=cd_other_file)
    train_sift_pool = Pool(train_sift_file, column_description=cd_other_file)
    train_orb_pool = Pool(train_orb_file, column_description=cd_other_file)

    for train_pool, iters, filename in ((train_comp_pool, 845, 'comp'), (train_surf_pool, 313, 'surf'),
                                        (train_sift_pool, 364, 'sift'), (train_orb_pool, 224, 'orb')):
        model = CatBoostClassifier(depth=3, iterations=iters, eval_metric='F1', task_type='CPU')
        model.fit(train_pool)
        model.save_model(filename + '.model')


def predict_catboost(id):
#    test_surf_file = '../surf/component_surf_IMG_' + id + '.df'
    test_comp_file = '../comp/component_IMG_' + id + '.df'
#    test_sift_file = '../sift/component_sift_IMG_' + id + '.df'
#    test_orb_file = '../orb/component_orb_IMG_' + id + '.df'

    cd_comp_file = '../components.cd'
#    cd_other_file = '../sift.cd'

    test_comp_pool = Pool(test_comp_file, column_description=cd_comp_file)
#    test_surf_pool = Pool(test_surf_file, column_description=cd_other_file)
#    test_sift_pool = Pool(test_sift_file, column_description=cd_other_file)
#    test_orb_pool = Pool(test_orb_file, column_description=cd_other_file)

#    surf_model = CatBoostClassifier().load_model('./surf.model')
    comp_model = CatBoostClassifier().load_model('./comp.model')
#    sift_model = CatBoostClassifier().load_model('./sift.model')
#    orb_model = CatBoostClassifier().load_model('./orb.model')
    preds = comp_model.predict(test_comp_pool)
    components = pbcomp.Components()
    write_preds_to_proto(components, preds, id)

'''    all_preds = []
    for (model, test_pool) in ((surf_model, test_surf_pool), (comp_model, test_comp_pool), (sift_model, test_sift_pool),
                  (orb_model, test_orb_pool)):
        preds = model.predict(test_pool)
        all_preds.append(preds)

    avg = np.zeros(len(all_preds[0]))
    avg = all_preds[1]
    for preds in all_preds:
        for i, pred in enumerate(preds):
            avg[i] += pred
            if avg[i] > 1:
                avg[i] = 1
    for i, pred in enumerate(avg):
        print (i, pred)'''


def write_preds_to_proto(components, preds, id):
    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "rb")
    components.ParseFromString(f.read())
    f.close()

    count = 0
    for comp in components.components:
        comp.pred = preds[count]
        count += 1
        print (comp.pred)

    f = open("/home/laneesra/CLionProjects/TextDetection/protobins/component_IMG_" + id + ".bin", "wb")
    f.write(components.SerializeToString())
    f.close()
