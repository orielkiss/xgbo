from config import cfg
import os
import numpy as np
from os.path import join
import uproot

from xgb_bo import XgbBoTrainer
import xgboost2tmva

out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

for idname in cfg["trainings"]:

    for training_bin in cfg["trainings"][idname]:

        print("Process training pipeline for {0} {1}".format(idname, training_bin))

        out_dir = join(out_dir_base, idname, training_bin)

        if not os.path.exists(out_dir):
            os.makedirs(join(out_dir))

        feature_cols = cfg["trainings"][idname][training_bin]["variables"]

        print("Reading data...")
        ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
        ntuple_file = join(ntuple_dir, 'train_eval.root')
        root_file = uproot.open(ntuple_file)
        tree = root_file["ntuplizer/tree"]

        df = tree.pandas.df(feature_cols + ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"], entrystop=None)

        df = df.query(cfg["selection_base"])
        df = df.query(cfg["trainings"][idname][training_bin]["cut"])
        df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

        print("Running bayesian optimized training...")
        xgb_bo_trainer = XgbBoTrainer(data=df, X_cols=feature_cols, y_col="y")
        xgb_bo_trainer.run()

        print("Saving weight files...")
        tmvafile = join(out_dir, "weights.xml")
        xgboost2tmva.convert_model(xgb_bo_trainer.models["bo"]._Booster.get_dump(),
                                   input_variables = list(zip(feature_cols, len(feature_cols)*['F'])),
                                   output_xml = tmvafile)
        os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
        os.system("mv {0} {0}.bak".format(tmvafile))
        os.system("mv {0}.tmp {0}".format(tmvafile))
        os.system("cd "+ out_dir + " && gzip -f weights.xml")

        print("Saving bayesian optimization results...")
        xgb_bo_trainer.get_results_df().to_csv(join(out_dir, "xgb_bo_results.csv"))

        print("Saving individual cv results...")
        if not os.path.exists(join(out_dir, "cv_results")):
            os.makedirs(join(out_dir, "cv_results"))
        for i, cvr in enumerate(xgb_bo_trainer.cv_results):
            cvr.to_csv(join(out_dir, "cv_results", "{0:04d}.csv".format(i)))

        print("Saving reduced data frame...")
        # Create a data frame with bdt outputs and kinematics to calculate the working points
        df_reduced = df.loc[xgb_bo_trainer.y_test.index,
                            ["ele_pt", "scl_eta", "y", "Fall17NoIsoV2RawVals", "genNpu"]]
        df_reduced["bdt_score_default"] = xgb_bo_trainer.get_score("default")
        df_reduced["bdt_score_bo"] = xgb_bo_trainer.get_score("bo")
        df_reduced.to_hdf(join(out_dir,'pt_eta_score.h5'), key='pt_eta_score')
