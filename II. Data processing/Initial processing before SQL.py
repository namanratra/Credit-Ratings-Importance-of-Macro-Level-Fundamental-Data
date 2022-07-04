import pandas as pd
import os
import numpy as np

### Load data
os.chdir("/Users/aayushmarishi/Desktop/Jamie/FS courses/Financial Management/Data processing/Documentation")
ratings = pd.read_csv('Ratings 2006-2015 original.csv')
economic = pd.read_csv('Economic data.csv')

### Create state initials-state name dictionary
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


### Clean up the column of state names in the economic file
economic['GeoName'] = economic['GeoName'].str.rstrip('* ')
economic = economic.drop('GeoFips', axis = 1)
economic['GDP_per_capita'] = economic['GDP'] / economic['Population']
economic = economic.drop(['GDP','Population'], axis =1)
economic.to_csv('Modified economic data.csv')

### Rename the ratings column and delete null values rows
ratings = ratings.rename(columns={'splticrm': 'Rating'})
ratings = ratings.dropna()
ratings = ratings.reset_index(drop=True)

### Convert the column of state intials in the rating file to align with the
### economic and weather data files
for i in range(204713):
    state = ratings['state'][i].strip()
    if state in states.keys():
        ratings['state'][i] = states[state]
    else:
        ratings = ratings.drop([i])

ratings.to_csv('Ratings data cleaned.csv')

### Extract gvkey from ratings file
gvkey = ratings['gvkey'].value_counts()
gvkey = pd.DataFrame(gvkey)
gvkey.reset_index(inplace=True)

gvkey_ = pd.DataFrame(gvkey['index'])
numpy_array = gvkey_.to_numpy()
np.savetxt("gvkey.txt", gvkey_, fmt = "%d")