import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ID from 1 to 20
student_ids = np.arange(1, 21)

# Random hours studied from 1 to 10
hours_stdied = np.random.randint(1, 11, size=20)
# Random exam scores between 50 and 100
exam_scores = np.random.randint(50, 101, size=20)

dataFrame = pd.DataFrame({
    'Student_ID': student_ids,
    'Hours_Studied': hours_stdied,
    'Exam_Score': exam_scores
})
print(dataFrame)


# List Function
def scores_above_70(scores_list):
    return [score for score in scores_list if score > 70]

# Tuple Function
def min_max_scores(scores_tuple):
    return (min(scores_tuple), max(scores_tuple))

# Dictionary Function
def student_score_dict(dataframe):
    return dict(zip(dataframe['Student_ID'], dataframe['Exam_Score']))

# Example usage:
to_scores_list = dataFrame['Exam_Score'].tolist()
to_scores_tuple = tuple(dataFrame['Exam_Score'])
to_score_dict = student_score_dict(dataFrame)

print("Scores above 70:", scores_above_70(to_scores_list))
print("Min and Max scores:", min_max_scores(to_scores_tuple))
print("Student ID to Exam Score mapping:", to_score_dict)

# Display summary statistics
print("\nSummary Statistics:")
print(dataFrame.describe())

# Sort by Exam_Score in descending order
sorted_dataFrame = dataFrame.sort_values(by='Exam_Score', ascending=False)
print("\nData sorted by Exam_Score (descending):")
print(sorted_dataFrame)