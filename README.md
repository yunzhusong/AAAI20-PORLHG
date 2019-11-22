# PORL-HG


## Dataset
CNNDM-DH, DM-DHC Datasets download link: [PORLHG](https://bit.ly/2TkSbIQ)

You can follow the [instructions](https://github.com/ChenRocks/cnn-dailymail) to download and preprocess the CNN/DailyMail dataset. 

The datasets can be associated by the **id**.

##
The source code will be updating in the coming future.

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
