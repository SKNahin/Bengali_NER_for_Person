# Bengali NER for Person
In this repository, Bengali NER Model is trained and evaluated using two datasets. Used datasets are - 
- [Dataset-1](https://github.com/Rifat1493/Bengali-NER/tree/master/annotated%20data)
- [Dataset-2](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl)

This raw datasets can be found in [data](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data) folder.

## Data Preprocessing
These datasets are not directly used for model training. As the datasets  contain multiple label at first,
the labels are transformed in binary form. Also, it was taken care of that the space separated parts of a sentence
align with the labels. If an word is a name or part of a name that is labeled 1 otherwise 0.

[Notebook](https://github.com/SKNahin/Bengali_NER_for_Person/blob/main/notebooks/processing_data.ipynb) for data preprocessing can be found in notebooks folder.

[processed_data](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data) folder contains [balanced](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data/processed_data/balanced) and [unbalanced](https://github.com/SKNahin/Bengali_NER_for_Person/tree/main/data/processed_data/unbalanced) folders. Both this folder contains train.json, valid.json and test.json files. In balanced folder train.json file contains some repeated examples of those sentences which contain at least one name. 

Here is the data distribution:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Distribution</th>
<th valign="bottom">Samples</th>
<!-- TABLE BODY -->
<!-- ROW: Train -->
 <tr><td align="left"></td>
<td align="center">1000</td>
</tr>
<!-- ROW: Valid -->
 <tr><td align="left"></td>
<td align="center">1000</td>
</tr>
<!-- ROW: Test -->
 <tr><td align="left"></td>
<td align="center">1000</td>
</tr>
</tbody></table>


Here is an example of finally used data for training-
```python
sentence = "আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম"
label =       1   1   0      0      0   0   0    0
```



  
  
