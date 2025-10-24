import matplotlib.pyplot as plt
import seaborn as sns



def check_dataset(df):
  print('Number of NAN values: \n', df.isna().sum())
  numeric_df = df.select_dtypes(include=['number'])
  print('\nMedian Value: \n', numeric_df.median())
  print('\nMode Value: \n', numeric_df.mode())
  print('\nStandard Deviation: \n', numeric_df.std())
  print('\nSkewness: \n', numeric_df.skew())

def plot_correlation(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues')
    plt.title('Correlation Matrix')
    plt.show()

def plot_feature_distributions(df):
    df.hist(figsize=(20,20))
    plt.show()