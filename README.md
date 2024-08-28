## Project Description: Empathy Assessment from Eye Fixations

This project explores the innovative use of eye tracking technology combined with machine learning algorithms to assess empathy levels based on eye fixations during social interactions. The primary objective is to develop a reliable and objective method for evaluating empathy, which can enhance social interactions, identify social deficiencies, and contribute to understanding the neurological underpinnings of empathy.

### Key Components

1. **Data Collection**: The dataset consists of eye movement data collected from participants during various tasks designed to elicit different levels of empathy. This includes a merged dataset that combines eye tracking data with participant metadata.

2. **Data Preprocessing**: The collected data undergoes rigorous preprocessing and cleaning to ensure accuracy and reliability. This includes checking for null values, converting categorical variables into numerical formats, and analyzing demographic information such as age and gender.

3. **Data Analysis**: Exploratory data analysis is conducted to identify patterns in eye movements, such as gaze duration and fixation counts, and to assess the relationship between these patterns and empathy scores.

4. **Machine Learning Modeling**: The project employs various machine learning algorithms, including Decision Trees and Neural Networks, to predict empathy levels based on eye fixation data. The models are evaluated for accuracy, precision, recall, and F1 scores, with the Decision Tree classifier demonstrating the highest accuracy of 97.8%.

5. **Results and Findings**: The findings indicate that eye fixations can serve as a valuable indicator of empathy, with the models successfully predicting empathy levels based on eye movement patterns. The project highlights the importance of using diverse datasets and robust methodologies to enhance model accuracy.

6. **Implications**: This research has significant implications across multiple fields, including psychology, psychiatry, and human-computer interaction. The insights gained can inform the development of tools and interventions aimed at improving social interactions and addressing social deficits.

### Conclusion

The project demonstrates the potential of using eye tracking and machine learning to provide an objective measure of empathy, paving the way for future research and applications in understanding human behavior and enhancing social outcomes.

### Code Overview

The provided code notebook covers the following key aspects of the project:

1. **Data Loading and Preprocessing**: The notebook starts by loading the necessary libraries and the merged dataset. It then proceeds to preprocess the data by checking for null values, converting categorical variables to numerical formats, and analyzing the dataset's shape and descriptive statistics.

2. **Exploratory Data Analysis**: The notebook performs extensive exploratory data analysis, including visualizing the distribution of gender, eye gaze classes, eye movement categories, tracking ratios based on gender and eye color, and generating a correlation heatmap.

3. **Machine Learning Modeling**: The notebook separates the independent and dependent variables, splits the data into training and testing sets, and applies cross-validation techniques. It then trains Decision Tree and Neural Network models, evaluating their performance using accuracy, precision, recall, F1 score, and cross-validation scores. The results show that the Decision Tree classifier achieves the highest accuracy of 97.8%.

4. **Model Comparison**: The notebook compares the performance of the Decision Tree and Neural Network models, highlighting the superior accuracy of the Decision Tree classifier.

5. **Discussion and Insights**: The notebook discusses the factors influencing the accuracy of the models, such as data quality, dataset diversity, and the choice of machine learning algorithms. It also provides insights into the potential applications and future directions of the project.

Overall, the code notebook provides a comprehensive implementation of the empathy assessment project, demonstrating the effectiveness of using eye tracking data and machine learning techniques to predict empathy levels.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/6289665/1a6077c4-275e-4a66-a8cf-7f7fa9e9d887/paste.txt
[2] https://drive.google.com/file/d/122pkK_3YeV6KvO_1nvjDaNuyA4N-9Ku9/view?usp=sharing
