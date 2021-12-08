#!/usr/bin/env python
# coding: utf-8

# # The Star Wars data 

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# The data used for the analysis is a dataframe containing all the characters in the chosen Star Wars movies, meaning all movies in the Skywalker saga, Star Wars 1-9. The dataframe is created such that we will get an easy overview for all characters and their associated attributes. The information of characters will come from the [Star Wars fandom wiki](https://starwars.fandom.com/wiki/Main_Page). There exists a unique wikipage for each movie, where all characters appearing in the movie will be listed. An example being:  
# <img src=data/wiki_appearances.PNG alt="wiki_appearances" width="500"/>  
# The characters in our dataframe will be from _Characters_ and _Creatures_ in the webpages. Furthermore, we will only extract the characters categorized as cannon, which means we will ignore legend characters.
# We have chosen to add _homeworld, species, gender, affiliation_ and _alliance_. 
# We have in total 1397 unique characters. 
# Here we an example of the dataframe. 

# In[2]:


characters_df = pd.read_csv('data/characters.csv')
characters_df[characters_df['Name'].isin(['Anakin Skywalker','R2-D2', 'Yané','Finn', 'Unidentified female First Order officer'])]

