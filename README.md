# PRODIGY_DS_02

## Prodigy InfoTech Data Science Internship – Task 02

### Task
Perform data cleaning and exploratory data analysis on the Titanic dataset.

### Dataset
Titanic passenger dataset.

Source:
https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

### Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn

### Data Cleaning

The following preprocessing steps were performed:

- Checked missing values.
- Filled missing `Age` values using the median age grouped by passenger class and gender.
- Filled remaining missing `Age` values using the overall median.
- Filled missing `Embarked` values using the mode.
- Removed the `Cabin` column because of extensive missing data.

### Exploratory Data Analysis

The analysis includes:

1. Survival rate by gender.
2. Survival rate by passenger class.
3. Correlation analysis of numerical variables.

### Key Findings

- The cleaned dataset contains 891 rows.
- Missing values were handled during preprocessing.
- Female passengers had a survival rate of 74.2%.
- Male passengers had a survival rate of 18.9%.
- First-class passengers had a survival rate of 63.0%.
- Third-class passengers had a survival rate of 24.2%.

### Outputs

The generated visualizations are available in the `outputs` folder:

- `survival_patterns.png`
- `correlation_heatmap.png`

### How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
