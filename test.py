from train import CCSBase2

ccs_model = CCSBase2("CCSMLDatabase.db",
                       "train_data.csv",
                       "test_data.csv",
                       n_estimators=6000,
                       max_depth=10,
                       learning_rate=0.03,
                       subsample=0.9,
                       colsample_bytree=0.9,
                       reg_lambda=30,
                       min_child_weight=5,
                       gamma=1,
                       seed=26,
                       use_metlin=True,
                       subclass_frequency_threshold=40,
                       fp_min_count=40
                    )
ccs_model.fit()
ccs_model.predict()