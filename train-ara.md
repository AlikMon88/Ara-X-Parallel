Training starts
        ↓
Ara agent monitors logs
        ↓
Issue detected (with patience threshold)
        ↓
Parallel automatically runs RCA
        ↓
Incident stored in memory
        ↓
Training continues

------------------------------

we train the model normally (model/sample_train_2.py) > ara watches parallely (runs on cloud) > it triggers parallel (with some patience) >> auto. sync like runner.py >> updates memory