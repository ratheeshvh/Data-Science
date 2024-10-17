import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("IMDbMoviesIndia.csv", encoding='latin-1')
data.head()

data.info()
data.isna().sum()

nulls = data[data.iloc[:, 1:9].isna().all(axis=1)]
nulls.head()

for col in data.select_dtypes(include = "object"):
    print(f"Name of Column: {col}")
    print(data[col].unique())
    print('\n', '-'*60, '\n')

data.dropna(subset=['Name', 'Year', 'Duration', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)
data['Name'] = data['Name'].str.extract('([A-Za-z\s\'\-]+)')
data['Year'] = data['Year'].str.replace(r'[()]', '', regex=True).astype(int)
data['Duration'] = pd.to_numeric(data['Duration'].str.replace(r' min', '', regex=True), errors='coerce')
data['Genre'] = data['Genre'].str.split(', ')
data = data.explode('Genre')
data['Genre'].fillna(data['Genre'].mode()[0], inplace=True)
data['Votes'] = pd.to_numeric(data['Votes'].str.replace(',', ''), errors='coerce')
duplicate = data.groupby(['Name', 'Year']).filter(lambda x: len(x) > 1)
duplicate.head(5)

data = data.drop_duplicates(subset=['Name'], keep=False)


data.describe()

data.describe(include = 'O')

max_votes_row = data[data['Votes'] == data['Votes'].max()]
movie_highest_votes = max_votes_row['Name'].values[0]
votes_highest_votes = max_votes_row['Votes'].values[0]
print("Movie with the highest votes:", movie_highest_votes)
print("Number of votes for the movie with the highest votes:", votes_highest_votes)
min_votes_row = data[data['Votes'] == data['Votes'].min()]
movie_lowest_votes = min_votes_row['Name'].values[0]
votes_lowest_votes = min_votes_row['Votes'].values[0]
print("Movie with the highest votes:", movie_lowest_votes)
print("Number of votes for the movie with the highest votes:", votes_lowest_votes)

max_rating_row = data[data['Rating'] == data['Rating'].max()]
movie_highest_rating = max_rating_row['Name'].values[0]
votes_highest_rating = max_rating_row['Votes'].values[0]
print("Movie with the highest rating:", movie_highest_rating)
print("Number of votes for the movie with the highest rating:", votes_highest_rating)
min_rating_row = data[data['Rating'] == data['Rating'].min()]
movie_lowest_rating = min_rating_row['Name'].values[0]
votes_lowest_rating = min_rating_row['Votes'].values[0]
print("Movie with the highest rating:", movie_lowest_rating)
print("Number of votes for the movie with the highest rating:", votes_lowest_rating)

director_counts = data['Director'].value_counts()
most_prolific_director = director_counts.idxmax()
num_movies_directed = director_counts.max()
print("Director with the most movies directed:", most_prolific_director)
print("Number of movies directed by", most_prolific_director, ":", num_movies_directed)
director_counts = data['Director'].value_counts()
least_prolific_director = director_counts.idxmin()
num_movies_directed = director_counts.min()
print("Director with the most movies directed:", least_prolific_director)
print("Number of movies directed by", most_prolific_director, ":", num_movies_directed)

colors = ['orange']
fig_year = px.histogram(data, x = 'Year', histnorm='probability density', nbins = 10, color_discrete_sequence = colors)
fig_year.update_traces(selector=dict(type='histogram'))
fig_year.update_layout(title='Distribution of Year', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Year', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), bargap=0.02, plot_bgcolor = 'white')
fig_year.show()

fig_rating = px.histogram(data, x = 'Rating', histnorm='probability density', nbins = 40, color_discrete_sequence = colors)
fig_rating.update_traces(selector=dict(type='histogram'))
fig_rating.update_layout(title='Distribution of Rating', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Rating', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), bargap=0.02, plot_bgcolor = 'white')
fig_rating.show()
fig_votes = px.box(data, x = 'Votes', color_discrete_sequence = colors)
fig_votes.update_layout(title='Distribution of Votes', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Votes', yaxis_title='Probability Density', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_votes.show()
average_rating_by_year = data.groupby('Year')['Rating'].mean().reset_index()
fig = px.line(average_rating_by_year, x='Year', y='Rating', color_discrete_sequence=['green'])
fig.update_layout(title='Are there any trends in ratings across year?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Year', yaxis_title='Rating', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig.show()
fig_dur_rat = px.scatter(data, x = 'Duration', y = 'Rating', trendline='ols', color = "Rating", color_continuous_scale = "darkmint")
fig_dur_rat.update_layout(title='Does length of movie have any impact on rating?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Duration of Movie in Minutes', yaxis_title='Rating of a movie', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_dur_rat.show()
fig_rat_votes = px.scatter(data, x = 'Rating', y = 'Votes', trendline='ols', color = "Votes", color_continuous_scale = "darkmint")
fig_rat_votes.update_layout(title='Does Ratings of movie have any impact on Votes?', title_x=0.5, title_pad=dict(t=20), title_font=dict(size=20), xaxis_title='Ratings of Movies', yaxis_title='Votes of movies', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), plot_bgcolor = 'white')
fig_rat_votes.show()