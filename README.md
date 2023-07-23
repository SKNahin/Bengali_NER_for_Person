# Bengali NER for Person
In this repository, Bengali NER Model is trained and evaluated using two datasets. Used datasets are - 
- [Dataset-1](https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data)
- [Dataset-2](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)

This raw datasets can be found in [data](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data) folder.

## Data Preprocessing
These datasets are not directly used for model training. As the datasets contain multiple labels, at first the labels are transformed in binary form. Also, it was taken care of that the space separated parts of a sentence align with the labels. If a word is a name or part of a name that is labeled `1` otherwise `0`.

[Notebook](https://github.com/SKNahin/Bengali_NER_for_Person/blob/main/notebooks/processing_data.ipynb) for data preprocessing can be found in notebooks folder.

[processed_data](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data) folder contains [balanced](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data/processed_data/balanced) and [unbalanced](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data/processed_data/unbalanced) folders. Both this folder contains `train.json`, `valid.json` and `test.json` files. In balanced folder `train.json` file contains some repeated examples of those sentences which contain at least one name. 

Here is the data distribution:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Distribution</th>
<th valign="bottom">Samples</th>
<!-- TABLE BODY -->
<!-- ROW: Train -->
 <tr><td align="left">Train</td>
<td align="center">7434</td>
</tr>
<!-- ROW: Valid -->
 <tr><td align="left">Valid</td>
<td align="center">929</td>
</tr>
<!-- ROW: Test -->
 <tr><td align="left">Test</td>
<td align="center">931</td>
</tr>
</tbody></table>


Here is an example of finally used data for training-
```python
sentence = "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম"
label =       1   1   0      0      0   0   0    0
```

## Train & Validation
Run [train.py](https://github.com/SKNahin/Bengali_NER_for_Person/blob/main/train.py) to train the model and run [test.py](https://github.com/SKNahin/Bengali_NER_for_Person/blob/main/test.py) to evaluate the model. `BanglaBert` is used here. Make sure to load the latest weight.

Also, you can run this [Notebook](https://github.com/SKNahin/Bengali_NER_for_Person/blob/main/notebooks/bengali-ner.ipynb) to train and evaluate the model.
The model is trained for three epochs, and the metric used to evaluate the model is the `F1 score`. The best `F1 score` was `0.8992`.

## Prediction
[predictions](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/predictions) folder contains `prediction.csv` file. Here is an example of model's prediction:

```python
sentence = "এরকম একজন হারিয়ে যাওয়া মনীষী কিশোরগঞ্জের আনন্দমোহন বসু ।"
prediction = "আনন্দমোহন বসু"
```

## Follow Paths
```python
Bengali_NER_for_Person/
|-- data/
|   |-- raw_data/
|   |-- processed_data/
|-- notebooks/
|   |-- bengali-ner.ipynb
|   |-- processing_data.ipynb
|-- predictions
|   |-- predictions.csv
|-- utils/
|   |-- helper.py
|   |-- preprocessor.py
|   |-- trainer.py
|-- requirements.txt
|-- train.py
|-- test.py

```
  
  
