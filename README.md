# Yelp Geolocation Project Repository
Group 5 - Zack Carlstrom, Daniel Evarone, Celia Cameron, Owen Brown, Alex Burde

## Dataset



## BERT

Implementation contained in (file name).

**Libraries**

- Transformers version: 4.51.3
- Datasets version: 3.5.0
- PyTorch version: 2.6.0+cu118
- scikit-learn version: 1.0.2
- Evaluate version: 0.4.3
- Pandas version: 1.4.4
- NumPy version: 1.21.5


**Hyperparameters**
    
    output_dir="./results",
    eval_strategy="steps",   
    eval_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=250,
    fp16=torch.cuda.is_available(),
    report_to="none"

**Final Results**

**Other notes**

### BERT Resources:
https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
https://huggingface.co/docs/transformers/en/index
https://huggingface.co/docs/datasets/en/index

## LSTM

Implementation contained in (file name). 

**Libraries**

- 

**Hyperparameters**

**Other notes**
