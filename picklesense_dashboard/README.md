# PickleSense Dashboard

Final deployed PickleSense dashboard project scaffold.

## Intended responsibilities

- Laptop camera + MediaPipe Pose
- Pose-TCN stroke classification
- MAX78000 / MPU6500 communication
- MAX78000 stroke prediction display
- Non-AI IMU statistics
- System status monitoring
- Unified live dashboard

## Model files to add later

Place the MediaPipe task model in:

    assets/models/pose_landmarker_lite.task

Place the trained pose model files in:

    trained_models/pose/picklesense_pose_tcn_final.pt
    trained_models/pose/pose_normalizer.json
