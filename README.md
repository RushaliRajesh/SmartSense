# SmartSense
Coding assessment 

The main.py file has all the code that was used in this assessment solution.
The model predicts whether the reviews are fake (deceptive) or truthful.

**Analysis of the dataset**
As part of analysing the dataset, many steps were carried out:
1. finding whether the dataset has any null values. And it was found that it didn't have any null values. This process in very crucial because later on when other steps of preprocessing or when the model training is carried out, an errorr will be encountered, which is definitely not desirable. Hence, checking for this null value is always important.

2. The dataset is balanced. There is no bias in regards to the number of samples from each class: deceptive and truthful.
   Both the classes have 800 samples each.

3. The 'hotels' column has 20 unique hotel names, and each of these hotels have 80 samples each.

4. There are 1600 rows (samples) and 5 columns(features) in the provided dataset.

**Preprocessing of the dataset**
1. Stopwords are removed from the "text" column in the dataset. Punctuation, articles(the,a,an), is, was are all examples of stopwords.
2. After getting the clean data, the contents of it are vectorized.
3. Tokenzing of the data in this column ("text") is carried out as well.
4. After getting the vectorized data of the "text" column, it is used for training the model.
5. The "deceptive" column in the dataset is then converted into a label encoding. (0 for truthfula nd 1 for deceptive/fake).

**Model Training and results**
1. Many ML algorithms can be used for this purpose like: SVM, decision trees etc. The one used here is logistic regression. Used logistic regression because it is jst a binary classification.
2. Initially only the vectorized "text" cloumn is used as feature matrix along with their respective labels as targets for modelling logistic regression.
3. The accuracy generated by this model was 85.75%.

**Extra/furthur implementations:**

Due to the time constraint it was only possible to train this model.
Also, as part of analysis of the dataset, we can also check for correlation between the dependent(feature) and independent columns(labels).
**But to get better and efficient results, it is better to use all the columns given in the dataset as features. BUT, we also need to check whether or not it is leading to some poor results due to "curse of dimensionality". Although the number of columns/features in this dataset is just 5, it is still advised and better to go with some dimensionality reduction techniques like: PCA, so that we get more optimised results.**
Also, there is an attached .png file which is just to demonstrate the dataset after the preprocessing.
