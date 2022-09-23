# %% importing libraries
import numpy as np
import pandas as pd
import datetime
import random

# %% date generator
def date_generator(date, month, year, number_of_dates):
    dates = []
    number_of_days = {1:31, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    for _ in range(number_of_dates):
        dates.append(f'{month}/{date}/{year}')
        
        date += 1
        leap_year = False
        if month == 2:
            if year % 4 == 0 or year % 100 == 0 :
                if year % 400 != 0 :
                    leap_year = True

        if month == 2:
            if leap_year:
                if date > 29 :
                    date = 1
                    month += 1
            else:
                if date > 28 :
                    date = 1
                    month += 1
        else:
            if date > number_of_days[month]:
                date = 1
                month += 1

        if month > 12:
            month = 1
            year += 1

    return dates
    
# %% 
def time(number_of_values):
    time = []
    l = ['AM', 'PM']
    for _ in range(number_of_values):
        hour = random.randint(1,12)
        minutes = random.randint(0,59)
        sec = random.randint(0,59)
        if hour < 10:
            hour = f'0{hour}'
        if minutes < 10:
            minutes = f'0{minutes}'
        if sec < 10:
            sec = f'0{sec}'
        time.append(f'{hour}:{minutes}:{sec} {random.choice(l)}')
    return time
# %% variables
data = pd.read_csv('myFile0.csv')

# %% attributes and methods note :- inplace is available every where
data.index
data.columns
data.axes
data.dtypes
data.shape
data.info()
data.fav_number
data.sum(axis=0)
data[['firstname', 'lastname']]
data['date'] = date_generator(1, 1, 2016, 1000)

data = pd.read_csv('myFile0.csv').dropna(how='all')
data.insert(0, column = 'date', value=date_generator(1, 1, 2016, 1000))

# %% broadcasting
data['scoring'].add(50)
data['scoring']+50
data['scoring'].sub(50)
data['scoring']-50
data['scoring'].mul(2)
data['scoring']*2

data.dropna()
data.drop(columns = ['address', 'scoring'], index = [2, 4, 6])

data['fav_number'].fillna(0) # for just one column
data.fillna(0) # for whole data set

def test(x):
    length = []
    for i in x:
        length.append(len(i))
    return length

data.sort_values('firstname', key = test)

data.sort_values(['firstname', 'lastname'], ascending=[True, False], na_position = 'first')
data['fav_number'].fillna(1, inplace=True)

data['fav_number_rank'] = data['fav_number'].rank(ascending=False).astype(int)
data.sort_values('fav_number_rank').head()


# %% data filtering
data["fav_number"] = data["fav_number"].fillna(0).astype("int")
data.sort_values("fav_number")
 
data[np.logical_or(data["fav_number"] > 500 , data["fav_number"] < 200)].head()
#other options and , np.logical_and , & , or , |

data[data["firstname"].isin(["Sara-Ann", "Margette", "Tani"])]

data = pd.read_csv('myFile0.csv')

data[data["fav_number"].isnull()]

data[data["fav_number"].notnull()]

data[data["fav_number"].between(200,800)]

data['date'] = date_generator(1, 1, 1990, 1000)
data['date'] = pd.to_datetime(data['date'])

data[data['date'].between(datetime.datetime(1990, 2, 1), datetime.datetime(1990, 10, 1))]

data[data['date'].between('1990-02-01', '1990-10-01')]

# print(time(100))


data['time'] = pd.to_datetime(time(1000))
# data.head()
data[data['time'].between('10:00AM', '12:00PM')]

# %% duplicate

data.sort_values(['firstname','lastname'], inplace=True)

data[data['firstname'].duplicated()]

data[np.logical_not(data['firstname'].duplicated())]

data[~data['firstname'].duplicated()]

data.drop_duplicates(['firstname'])

# %% indexing
data = pd.read_csv('myFile0.csv')
data['date'] = date_generator(1, 1, 1990, 1000)
data['date'] = pd.to_datetime(data['date'])
data['time'] = pd.to_datetime(time(1000))

data.set_index("id", inplace=True)
data.sort_index(inplace=True)
data.head()

data.loc[100, 'firstname']

data.loc[100:200] # slicing 


data.loc[[100, 200]]

data.loc[[100, 1000]]

# same as above just uses index and for slicing last index not included
data.iloc[0:1000:100]['firstname']

data.loc[100, ['firstname', 'lastname']]

data.loc[100, 'firstname'] = 'SSara-Ann'
data.loc[100, 'firstname'] #doesn't work on iloc it makes a copy

data.loc[100, 'firstname'] = 'Sara-Ann'
data.loc[100, 'firstname']

data.rename(columns={'firstname': 'Firstname', 'lastname': 'Lastname'}, inplace=True)
data.head()

data.rename(index={100:99}, inplace=True)
data.head()

data.rename(index={99:100}, inplace=True)
data.head()

data.rename(columns={'Firstname': 'firstname', 'Lastname': 'lastname'}, inplace=True)
data.head()
# %% 
data = pd.read_csv('myFile0.csv')
data['date'] = date_generator(1, 1, 1990, 1000)
data['date'] = pd.to_datetime(data['date'])
data['time'] = pd.to_datetime(time(1000))
data.set_index("id", inplace=True)
data.sort_index(inplace=True)

data.query('firstname == "Sara-Ann"')

data.query('lastname == "Margret"')

data.query('id != 101')

data.query('id > 500')

data.query('id > 500 and firstname == "Sara-Ann"')

data.query('firstname in ["Sara-Ann", "Aaren"]')

data.drop(100) # row with id 100

data.drop(['firstname', 'lastname'], axis=1, inplace=False)

Firstname = data.pop('firstname')
data
Firstname

# %% 
data = pd.read_csv('myFile0.csv')
data['date'] = date_generator(1, 1, 1990, 1000)
data['date'] = pd.to_datetime(data['date'])
data['time'] = pd.to_datetime(time(1000))
data.set_index("id", inplace=True)
data.sort_index(inplace=True)

data.sample(n=100)
data.sample(frac=.25)
data.sample(n = 3, axis=1)

data.nsmallest(10, 'fav_number')
data.nlargest(10, 'fav_number')

data['fav_number'].nlargest(10)

data[np.logical_and(data['firstname']=='Sara-Ann',data['lastname']=='Wyn')]
data.where(data['fav_number']>800).dropna(how='all')

def multiply_100(number):
    return number*100

data['fav_number'].apply(multiply_100)

def best_person(row):
    
    score = row[2]
    fav_number = row[4]

    if score > 800:
        return 'Great person'
    
    elif score > 500 and fav_number > 500:
        return 'Nice person'
    
    else:
        return "I don't know"

data.apply(best_person, axis='columns')

