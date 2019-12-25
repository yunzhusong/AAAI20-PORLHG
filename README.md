# PORL-HG


## Dataset
CNNDM-DH, DM-DHC Datasets download link: [PORLHG](https://bit.ly/2TkSbIQ)

You can follow the [instructions](https://github.com/ChenRocks/cnn-dailymail) to download and preprocess the CNN/DailyMail dataset to acquire the **article**.

The dataset is collected according to the url links provided by <cite>[Nallapati et al. 2016](https://arxiv.org/abs/1602.06023)</cite>, <cite>[Hermann et al. 2015](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend.pdf)

The DH, DHC datasets can be associated with CNNDM by the **id**.

The dataset information:

|    | train  | val   | test  |
|----|--------|-------|-------|
| DH | 281208 | 12727 | 10577 |
| DHC| 138787 | 11862 | 10130 |


##
The source code will be updated in the coming future.

## More Experiment Results

Table1. Correlation Analysis of CTR, comments and shares
List of hypotheses and the corresponding p-value of the significance test, where **bold text** indicates significant hypothesis (p-value < 0.05). Note the p-value of CTR is referenced from <cite>[Kuiken et al. 2017](https://www.tandfonline.com/doi/full/10.1080/21670811.2017.1279978)</cite>

|Hypothesis | CTR | Comment | Share|
| ------------- |-------------| -----| -----|
|H1 Longer headline(> 50 characters) are preferred over shorter headlines|0.297|**0**|**0**|
|H2 Headlines with short words (< 8 characters per word) are preferred |**0.024**|**0**|**0**|
|H3 Headlines containing a question are preferred|**0.019**|**0**|**0**|
|H4 Headlines containing a partial quote are preferred over not containing any quote|0.239|0.996|0.971|
|H5 Headlines not containing any quote are preferred over containing full quote|**0.03**|0.848|0.111|
|H6 Headlines that contain one or more signal words are preferred |**0.002**|**0**|**0.001**|
|H7 Headlines that contain one or more personal or possessive pronouns are preferred|**0**|**0**|**0**|
|H8 Headlines that contain one or more sentimental words are preferred|**0.018**|**0**|**0**|
|H9 Headlines that contain one or more negative sentimental word are preferred|**0.001**|**0.001**|**0.015**|
|H10 Headlines that contain a number are preferred over headlines that do not|0.202|**0**|**0.06**|
|H11 Headlines that start with a personal or possessive pronoun are preferred|**0.002**|**0**|0.429|



Table2. The popularity features. The following 11 features are transformed from the hypotheses stated in Table1. GT indicates the abbreviation of ground-truth headlines, and Chen et al. is one of our baselines <cite>[Chen et al. 2018](https://arxiv.org/abs/1805.11080)</cite>.

|Hypothesis|Significance|GT|PORL|Chen et al.|
|----|----|----|----|----|
|H1 The average character length of a headline | False| 70.55|**96.21**|73.92|
|H2 The average of token lengths in a headline (lower is better) | True |  4.97 | **4.78** | 4.89|
|H3 The percentage of headlines containing a question mark | True | **2.52** | 0.90% | 1.19%|
|H4 The percentage of headlines containing a partial quote | True | 11.81% |**15.80%** | 13.85%|
|H5 The percentage of headline containing full quote (lower is better) | False | 0.01% | **0.00%** | **0.00%** |
|H6 The percentage of headline containing signal words | True | 9.90% | **19.83%** | 15.00% |
|H7 The percentage of headline containing personal or possessive pronoun | True | 28.82% | **48.67%** | 40.35%|
|H8 The percentage of headline containing sentimental words | True | 68.82% | **77.40%** | 69.37% |
|H9 The percentage of headline containing negative words | True | 45.09% | **52.29%** | 44.83%|
|H10 The percentage of headline containing numbers | False | 20.58% | **25.22%** | 21.06% |
|H11 The percentage of headline starting with personal or possessive pronoun | True | 0.64% |**1.07%** | 0.38% |
