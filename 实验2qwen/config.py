class Config:
    model_name = "Qwen/Qwen2.5-0.5B"

    max_seq_length = 128
    num_classes = 2
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10
    warmup_ratio = 0.1

    model_save_path = "saved_models/qwen_model.pth"

    train_path = "dataset/train_filtered.csv"
    dev_path = "dataset/dev.csv"
    test_path = "dataset/test.csv"
