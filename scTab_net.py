import subprocess

#TODO: change fixed parameters to be passed as arguments
def train_and_evaluate_scTab(
    adata_train,
    adata_test, 
    label_key, 
    label_encoder, 
    num_classes,      
    epoch_num, 
    device,         
    batch_size
):
    """
    Calls the TabNet training script with the specified arguments.
    """
    print("Starting TabNet training...")

    cmd = [
        "python3", "-u",
        "/cs/usr/idan724/lab/scTab/scripts/py_scripts/train_and_eval_tabnet.py",
        "--cluster=jsc",
        "--version=version_1",
        "--data_path=/cs/usr/idan724/lab/merlin_cxg_minimal/merlin_cxg_2023_05_15_sf-log1p_minimal",
        "--epochs=30",
        "--batch_size=4096",
        "--sub_sample_frac=1.0",
        "--lr=0.005",
        "--weight_decay=0.05",
        "--use_class_weights=True",
        "--lambda_sparse=1e-5",
        "--n_d=32",
        "--n_a=32",
        "--n_steps=1",
        "--gamma=1.3",
        "--n_independent=1",
        "--n_shared=1",
        "--virtual_batch_size=1024",
        "--mask_type=entmax",
        "--augment_training_data=True",
        "--lr_scheduler_step_size=1",
        "--lr_scheduler_gamma=0.9",
        "--check_val_every_n_epoch=1",
        "--checkpoint_interval=1",
        "--seed=1"
    ]

    # Run the command
    subprocess.run(cmd, check=True)
    print("TabNet training script completed.")


