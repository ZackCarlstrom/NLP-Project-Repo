# Yelp Geolocation Project Repository
Group 5 - Zack Carlstrom, Daniel Evarone, Celia Cameron, Owen Brown, Alex Burde

## Dataset & Data Exploration

The dataset we used comes from the Yelp Open Dataset: https://business.yelp.com/data/resources/open-dataset/

If you’d like to go through the data processing and exploration file, you can download the JSON data by following the link above. 

The following filtered datasets were used in the project:
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

Implementation contained in **Naive_Bayes_Yelp.ipynb**.

**Libraries**

- Pandas version: 2.2.2
- Scikit-learn version: 1.4.2

**Hyperparameters**

- test_size = 0.2
- random_state = 40
- stop_words = "english"
- max_features = 20000

**Results**
- 2 regions: average precision of 0.50, average recall of 0.50, average F1 of 0.50
- 4 regions: average precision of 0.50, average recall of 0.49, average F1 of 0.49
- 13 states: average precision of 0.56, average recall of 0.11, average F1 of 0.09

A multinomial Naive Bayes classifier was developed and used on the processed data. This was to be used as a baseline to explore the validity of researching this problem. As a fairly simple model, it did not perform very well, with performance getting worse the more complexity was introducted. It was clearly very vulnerable to the varieties in the data and was not suitable to account for the complexities of the problem.

### Naive Bayes Resources:

https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html


## BERT

Implementation contained in **BERT Pretraining & Evaluation.ipynb**. After fine-tuning and evaluation, each model (at best validation f1-score) and tokenizer is saved under local directories specified in the notebook.

**Libraries**

- Transformers version: 4.51.3
- Datasets version: 3.5.0
- PyTorch version: 2.6.0+cu118
- scikit-learn version: 1.0.2
- Evaluate version: 0.4.3
- Pandas version: 1.4.4
- NumPy version: 1.21.5

**Hyperparameters**

- output_dir="./results",
- eval_strategy="steps",
- eval_steps=250,
- load_best_model_at_end=True,
- metric_for_best_model="f1",
- num_train_epochs=5,
- learning_rate=2e-5,
- per_device_train_batch_size=16,per_device_eval_batch_size=16,
- weight_decay=0.01,
- save_total_limit=1,
- logging_dir="./logs",
- logging_steps=250,
- fp16=torch.cuda.is_available(),
- report_to="none"

**Results**
- 4 regions: average precision of 0.51, average recall of 0.50, average F1 of 0.51
- 4 regions w/prompting: average precision of 0.51, average recall of 0.50, average F1 of 0.50
- 13 states: average precision of 0.38, average recall of 0.36, average F1 of 0.36

Fine-tuning the BERT-base-uncased model showed minimal improvement over Bayes in the 4 region tasks, but showed a large improvement in F1-score for classifying the 13 states. Prompting in the dataset was used in one trial, but seemed to have no significant impact, getting a F1-score 0.01 lower than the same task without prompting. Compared to the LSTM models, the fine-tuned BERT model had a slightly higher average F1-score.

### BERT Resources:
https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c

https://huggingface.co/docs/transformers/en/index

https://huggingface.co/docs/datasets/en/index

## RoBERTa

Implementation contained in **RoBERTa Pretraining & Evaluation.ipynb**. After fine-tuning and evaluation, each model (at best validation f1-score) and tokenizer is saved under local directories specified in the notebook.

**Libraries**

- Transformers version: 4.51.1
- Datasets version: 3.5.0
- PyTorch version: 2.1.0+cu118
- scikit-learn version: 1.3.2
- Evaluate version: 0.4.3
- Pandas version: 2.2.3
- NumPy version: 1.26.4

**Hyperparameters**

- output_dir="./results-roberta-base",
- eval_strategy="steps",
- eval_steps=250,
- load_best_model_at_end=True,
- metric_for_best_model="f1",
- num_train_epochs=5,
- learning_rate=2e-5,
- per_device_train_batch_size=16,
- per_device_eval_batch_size=16,
- weight_decay=0.01,
- save_total_limit=1,
- logging_dir="./logs-roberta-base",
- logging_steps=250,
- fp16=torch.cuda.is_available(),
- report_to="none"

**Results**
- 4 regions: average precision of 0.51, average recall of 0.50, average F1 of 0.50
- 4 regions w/prompting: average precision of 0.51, average recall of 0.50, average F1 of 0.50
- 13 states: average precision of 0.41, average recall of 0.35, average F1 of 0.36

Fine-tuning the RoBERTa-base model showed similar results to the fine-tuned BERT-base-uncased model. There was once again minimal improvement over Bayes in the 4 region tasks, but the model showed a large improvement in F1-score for classifying the 13 states. Fine-tuning with hard prompting also showed no change in overall performance. The RoBERTa fine-tuned model still slightly out performed the LSTM models with a slightly higher F1-score. For training time, RoBERTa took slightly longer than the BERT models, but only by a slim margin.

## LSTM

Implementation contained in **Final_Project_Yelp v2.ipynb** and **Final_Project_Yelp v2 Flags.ipynb**. Any of the tests below with flags are in the Flag specific file.

**Libraries**

- Python: 3.10.11
- Torch: 2.1.0+cu118
- NumPy: 1.26.4
- Pandas: 2.2.3
- scikit-learn: 1.3.2
- NLTK: 3.9.1
- Matplotlib: 3.10.0

**Hyperparameters**

These are the hyperparameters for the highest performing model, the 200k samples, replacement, flags, 5 epochs model with an average F1 of 0.29.

- lstm_hidden_size=50
- num_lstm_layers=1
- bidirectional=True
- learning_rate=0.001
- batch_size=128
- epochs=10
- tokenized=False
- n_samples=100000 (reviews per state)
- num_labels=13
- test_size=0.2

**Results**

All of these were tested with 13 states, instead experimenting with the number of samples and modifications like tokenizing, removing stopwords, and flagging.

- 50k samples, tokenized: average precision of 0.32, average recall of 0.28, average F1 of 0.27
- 50k samples: average precision of 0.29, average recall of 0.24, average F1 of 0.24
- 50k samples, tokenized, stopwords removed: average precision of 0.29, average recall of 0.25, average F1 of 0.25
- 200k samples, replacement: average precision of 0.33, average recall of 0.29, average F1 of 0.28
- 200k samples, replacement, flags, 2 epochs: average precision of 0.30, average recall of 0.27, average F1 of 0.27
- 200k samples, replacement, flags, 5 epochs: average precision of 0.31, average recall of 0.29, average F1 of 0.29
- 100k samples per state, replacement, 10 epochs: average precision of 0.34, average recall 0.32, average F1 of 0.31

Louisiana had the highest F1 score overall across all testing, so we investigated the reviews further. An analysis of the top 200 correctly classified reviews found that a majority of them contained “New Orleans,” so an idea was hatched to create a dictionary of the 10 largest or most popular cities in each state in the dataset, as well as 4 landmarks in that state. This seemed to improve the results slightly, enough to leave the flags included for future tests. More tests were run with different ways of samping from the dataset, including pulling 50,000 reviews from each state or using up to 200,000 reviews from each state with replacement. These methods produced balanced training and test sets, where the previous iterations used the entire dataset that was unbalanced. We experimented with different epoch sizes, with 2 not learning enough and 50 overfitting too much to be helpful. While 5 was enough to get a good fit, we improved the model by 2% F1 score by increasing to 10 epochs. We found the best results for accuracy and runtime were 50k to 100k reviews per state with 5 to 10 epochs. Increasing these parameters did not increase the F1 score in a meaningful way.

Overall, we found that the models struggled with geographical classification for Yelp reviews even through training and other optimization methods. This may be due to written language not being conducive to inflection or the probability that Yelp reviews are not necessarily authored by residents of those states or cities.

### Contributions
- Zack Carlstrom:  BERT, *Lead*; RoBERTa, *Lead*; Data Exploration and Manipulation, *Lead*; Written Deliverables, *Contributor*

- Daniel Evarone: LSTM, *Lead*; Error Analysis, *Lead*

- Celia Cameron: Naive Bayes, *Lead*; Presentation, *Lead*; Written Deliverables, *Contributor*

- Owen Brown: Presentation, *Contributor*; Written Deliverables, *Contributor*

- Alex Burde: Presentation, *Contributor*
