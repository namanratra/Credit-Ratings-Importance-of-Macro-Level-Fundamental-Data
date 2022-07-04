import pandas as pd
import os
import numpy as np

# Load data
os.chdir("/Data processing/Documentation")
data = pd.read_csv('Rating_fiscal_econ_ratio.csv')
weather = pd.read_csv('Weather data 2006-2015.csv')

# convert damage property  number and damage crops number
weather['DAMAGE_PROPERTY'] = weather['DAMAGE_PROPERTY'].fillna(0)
weather['DAMAGE_CROPS'] = weather['DAMAGE_CROPS'].fillna(0)
weather['DAMAGE_PROPERTY1']  = weather['DAMAGE_PROPERTY'].apply(lambda x: str(x)[-1])
weather['DAMAGE_CROPS1']  = weather['DAMAGE_CROPS'].apply(lambda x: str(x)[-1])
weather['DAMAGE_PROPERTY1'].unique()
weather['DAMAGE_CROPS1'].unique()

def number_transfer(x):
    if str(x)[-1] == 0 :
        x = float(x)
    elif str(x) =='K':
        x = 1000
    elif str(x)[-1] == 'K':
        x = 1000* float(x[:-1])
    elif str(x) == 'M':
        x = 1000000
    elif str(x)[-1] == 'M':
        x = 1000000 * float(x[:-1])
    elif str(x) == 'B':
        x = 1000000000
    elif str(x)[-1] == 'B':
        x = 1000000000 * float(x[:-1])
    return float(x)

# convert columns "damage_property" and "damage_crops", both of which contains
# string to express magnitude (e.g. "0.5k") into numerical values
weather['DAMAGE_PROPERTY1'] = weather['DAMAGE_PROPERTY'].apply(lambda x: number_transfer(x))
weather['DAMAGE_CROPS1']  = weather['DAMAGE_CROPS'].apply(lambda x: number_transfer(x))


# convert tornado scales into numbers
tornado_scale = {'EF0':1,'EF1':2, 'EF2':3, 'EF3':4, 'EF4':5, 'EF5':6,
                 'F0':1,'F1':2, 'F2':3, 'F3':4, 'F4':5,np.nan: 0}
weather['TOR_F_SCALE1'] = weather['TOR_F_SCALE'].map(tornado_scale)

# filter to include only observations related to the six weather events we
# are interested in
event = ['Flood', 'Tornado', 'Tropical Storm','Heat','Winter Storm', 'Wildfire']
weather['filter_'] = weather['EVENT_TYPE'].apply(lambda x : True if x in event else False)
weather1 = weather.loc[weather['filter_'] == True]
weather1.dtypes

weather2 = weather1[['EVENT_ID', 'STATE', 'YEAR','EVENT_TYPE',
       'INJURIES_DIRECT', 'DEATHS_DIRECT', 'DAMAGE_PROPERTY1', 'DAMAGE_CROPS1',
       'TOR_F_SCALE1']]
weather2.columns = ['EVENT_ID', 'STATE', 'YEAR','EVENT_TYPE',
       'INJURIES_DIRECT', 'DEATHS_DIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS',
       'TOR_F_SCALE']

# calculate every calculation method of every event 
aggfunc = {'INJURIES_DIRECT':np.sum,'DEATHS_DIRECT':np.sum,'DAMAGE_PROPERTY':np.sum,
           'DAMAGE_CROPS':np.sum,'TOR_F_SCALE' :np.mean,'EVENT_ID':np.count_nonzero}

# a function to pivot with specific parameter and ravel index 
def event_stat(df,values,columns):
    df1 = pd.pivot_table(df,values = values,columns = columns,index = ['YEAR','STATE']
    ,aggfunc = aggfunc[values])
    df1 = df1.reset_index()
    df1.columns =[''.join(x) for x in df1.columns.ravel()]
    return df1 

#create pivot table from the weather dataset before merging
event_count = event_stat(weather2,values = 'EVENT_ID',columns = 'EVENT_TYPE')

damage_crops_sum = event_stat(weather2, values = 'DAMAGE_CROPS',columns = 'EVENT_TYPE')

damage_property_sum = event_stat(weather2, values = 'DAMAGE_PROPERTY',columns = 'EVENT_TYPE')

injury_sum = event_stat(weather2, values = 'INJURIES_DIRECT',columns = 'EVENT_TYPE')

death_sum = event_stat(weather2, values = 'DEATHS_DIRECT',columns = 'EVENT_TYPE')

tornado_scale_avg = event_stat(weather2, values = 'TOR_F_SCALE',
                               columns = 'EVENT_TYPE')[['STATE','YEAR','Tornado']]

data = data.drop('ratio_year', axis = 1)

#creating new columns in the dataset to store weather-related data
data['flood_count'] = 0
data['heat_count'] = 0
data['tornado_count'] = 0
data['tropical_storm_count'] = 0
data['wildfire_count'] = 0
data['winter_storm_count'] = 0

data['flood_crop'] = 0
data['heat_crop'] = 0
data['tornado_crop'] = 0
data['tropical_storm_crop'] = 0
data['wildfire_crop'] = 0
data['winter_storm_crop'] = 0

data['flood_property'] = 0
data['heat_property'] = 0
data['tornado_property'] = 0
data['tropical_storm_property'] = 0
data['wildfire_property'] = 0
data['winter_storm_property'] = 0

data['flood_injury'] = 0
data['heat_injury'] = 0
data['tornado_injury'] = 0
data['tropical_storm_injury'] = 0
data['wildfire_injury'] = 0
data['winter_storm_injury'] = 0

data['flood_death'] = 0
data['heat_death'] = 0
data['tornado_death'] = 0
data['tropical_storm_death'] = 0
data['wildfire_death'] = 0
data['winter_storm_death'] = 0



data['avg_tornado_scale'] = 0

# merging with weather data
pivot_tables = [event_count, damage_crops_sum, damage_property_sum, injury_sum,
                death_sum, tornado_scale_avg]

for i in range(10563):
    s = data['state'][i]
    y = data['year'][i]
    for w in range(5):
        table = pivot_tables[w]
        for e in range(6):
            v = table.loc[table['YEAR'] == y].loc[table['STATE'] == s.upper()].iloc[:,2+e]
            value = list(v)[0]
            col = 11 + w * 6 + e
            data.iloc[i,col] = value
    sc = tornado_scale_avg.loc[tornado_scale_avg['YEAR'] == y].loc[tornado_scale_avg['STATE'] == s.upper()].iloc[:,2]
    scale = list(sc)[0]
    data.iloc[i,-1] = scale     

# creating new columns of dummy variable for each weather event to indicate
# their existence
data['flood_dummy'] = (data['flood_count'] > 0) * 1
data['heat_dummy'] = (data['heat_count'] > 0) * 1
data['tornado_dummy'] = (data['tornado_count'] > 0) * 1
data['tropical_storm_dummy'] = (data['tropical_storm_count'] > 0) * 1
data['wildfire_dummy'] = (data['wildfire_count'] > 0) * 1
data['winter_storm_dummy'] = (data['winter_storm_count'] > 0) * 1

# fill in NaN values and move the target column (rating) to the last position
data_2 = data.fillna(0)
target = "Rating"
last_col = data_2.pop(target)
data_2.insert(47, target, last_col)

# convert credit rating into numerical values (aggregated level)
sp_rating = {"AAA": 1, "AA+": 2, "AA": 2, "AA-": 2, "A+": 3, "A": 3, "A-": 3,
             "BBB+": 4, "BBB": 4, "BBB-": 4, "BB+": 5, "BB": 5, "BB-": 5,
             "B+": 6, "B": 6, "B-": 6, "CCC+": 7, "CCC": 7, "CCC-": 7,
             "CC": 8, "D": 9, "SD": 9}
data_2['Rating_num'] = data_2['Rating'].map(sp_rating)
data_2 = data_2.drop('Rating', axis = 1)

data_2 = data_2.rename(columns={'conml': 'Company'})
data_2.to_csv('Final rating dataset.csv')
