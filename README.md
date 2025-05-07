# Yelp Geolocation Project Repository
Group 5 - Zack Carlstrom, Daniel Evarone, Celia Cameron, Owen Brown, Alex Burde

## Dataset & Data Exploration

The dataset we used comes from the Yelp Open Dataset: https://business.yelp.com/data/resources/open-dataset/

If you’d like to go through the data processing and exploration file, you can download the JSON data by following the link above. 

The following datasets filtered data sets were used in the project:
- Full Filtered Dataset (review, state, region): https://drive.google.com/file/d/1NfMHyc4aGIHF1h-svRQDAbVkK_q3_-do/view?usp=sharing
- Balanced Region Dataset (10k per region w/ reviews): https://drive.google.com/file/d/1qoy4TU2nleo8Z0AW4wwbrxA2l08u7c5Z/view?usp=sharing
- Balanced State Dataset (5k per state w/ reviews): https://drive.google.com/file/d/1_49LMdYTZNSSJH_0i8WyUGk0jJiZOmfi/view?usp=sharing

Where to use these Datasets: 
- The Full Filtered Dataset is used in the BERT, RoBERTa, and LSTM models for training and testing. 
- The balanced region and state datasets are used in the Naive Bayes model due to hardware limitations for certain team members, which prevented them from loading the full filtered dataset.

Dataset Notes:
- When using these datasets, you will have to update the code to contain the location of the compressed csv files in your file system. 

Libraries Needed for Data Exploration Notebook:
- Altair version: 5.5.0
- Pandas version: 2.2.3
- JSON & CSV standard modules



## Naive Bayes

Implementation contained in (file name).

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


### Contributions
- Zack Carlstrom:  BERT, (Lead); Data Exploration and Manipulation, (Lead); Naive Bayes, *Contributor*; Written Deliverables, *Contributor*

- Daniel Evarone: LSTM, *Lead*; Error Analysis, *Lead*

- Celia Cameron: Naive Bayes, *Lead*; Presentation, *Lead*; Written Deliverables, *Contributor*

- Owen Brown: Presentation, *Contributor*; Written Deliverables, *Contributor*

- Alex Burde: Presentation, *Contributor*
