import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import geopandas
import plotly.express as px
from datetime import datetime

st.set_page_config(layout='wide')

# functions
## read data
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    return data

@st.cache(allow_output_mutation=True)
def descriptive_statistics(data):
    '''
    Return numeric variables dataframe statistics, including min, max, mean, median and std.
    '''
    df_statistics = data.describe().T.reset_index().rename({'index':'attributes', 
                                                            '50%':'median'}, axis=1)
    df_statistics = df_statistics[['attributes', 'min', 'max', 
                                   'mean', 'median', 'std']]
    return df_statistics

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

# load data
path = 'C:/Users/Emerson/Documents/Jupyter Projects/pthon_do_zero_ao_ds/datasets/kc_house_data.csv'
df = get_data(path)

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# add new features
## price by square meters
df['price_m2'] = round(df['price'] / (df['sqft_lot'] / 10.764), 2)

## living room in square meters
df['living_m2'] = round(df['sqft_living'] / 10.764, 2)

# data overview
f_attributes = st.sidebar.multiselect('Enter columns', df.columns)
f_zipcode = st.sidebar.multiselect('Enter zipcode', df['zipcode'].unique())

st.title('Data Overview')

if (f_zipcode != []) & (f_attributes != []):
    df = df.loc[df['zipcode'].isin(f_zipcode), f_attributes]
elif (f_zipcode != []) & (f_attributes == []):
    df = df.loc[df['zipcode'].isin(f_zipcode), :]
elif (f_zipcode == []) & (f_attributes != []):
    df = df.loc[:, f_attributes]
else:
    df = df.copy()

st.dataframe(df.head())

# new dataframes
c1, c2 = st.beta_columns((1, 1))

## descriptive statistics
stat_df = descriptive_statistics(df)

c1.header('Descriptive Statistics')
c1.dataframe(stat_df, height=600)

## average metrics
df1 = df[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = df[['living_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = df[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

### merge
m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
avg_df = pd.merge(m2, df4, on='zipcode', how='inner')

avg_df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'LIVING ROOM M2', 'PRICE/LOT M2']

c2.header('Average Values by Zipcode')
c2.dataframe(avg_df, height=575)

# portfolio density
""" st.title('Region Overview')

c1, c2 = st.beta_columns((1, 1))
c1.header('Portfolio Density')

## base map - folium
density_map = folium.Map(location=[df['lat'].mean(), df['long'].mean()],
                         default_zoom_start=15)

marker_cluster = MarkerCluster().add_to(density_map)
for name, row in df.iterrows():
    folium.Marker([row['lat'], row['long']],
                  popup='Sold ${0} on: {1}. Features: {2} m2, {3} bedrooms, {4} bathrooms, year built: {5}.'.format(row['price'],
                                                                                                                    row['date'],
                                                                                                                    row['living_m2'],
                                                                                                                    row['bedrooms'],
                                                                                                                    row['bathrooms'],
                                                                                                                    row['yr_built'])).add_to(marker_cluster)

with c1:
    folium_static(density_map)

## region price map
c2.header('Price Density')

df_aux = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df_aux.columns = ['ZIP', 'PRICE']

geofile = geofile[geofile['ZIP'].isin(df_aux['ZIP'].tolist())]

region_price_map = folium.Map(location=[df['lat'].mean(), df['long'].mean()],
                              default_zoom_start=15)

folium.Choropleth(data=df_aux, geo_data=geofile, 
                  columns=['ZIP', 'PRICE'], 
                  key_on='feature.properties.ZIP',
                  fill_color='YlOrRd',
                  fill_opacity=0.7,
                  line_opacity=0.2,
                  legend_name='AVG PRICE').add_to(region_price_map)

with c2:
    folium_static(region_price_map) """

# house distribution per commercial attributes
st.sidebar.title('Commercial Options')
st.title('Commercial Attributes')

## average price per year
st.header('Average Price per Year Built')
st.sidebar.subheader('Select Max Year')

### filters
min_year_built = df['yr_built'].min()
max_year_built = df['yr_built'].max()

f_yr_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, max_year_built)

### data selection
yr_df = df.loc[df['yr_built'] <= f_yr_built] 
avg_price_yr_df = yr_df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

### plot
fig = px.line(avg_price_yr_df, x='yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)

## average price per day
st.header('Average Price per Day')
st.sidebar.subheader('Select Max Date')

### filters
min_date = datetime.strptime(df['date'].min().strftime('%Y-%m-%d'), '%Y-%m-%d')
max_date = datetime.strptime(df['date'].max().strftime('%Y-%m-%d'), '%Y-%m-%d')

f_date = st.sidebar.slider('Date', min_date, max_date, max_date)

### data selection
date_df = df.loc[df['date'] <= f_date]
avg_price_day_df = date_df[['date', 'price']].groupby('date').mean().reset_index()

### plot
fig = px.line(avg_price_day_df, x='date', y='price')
st.plotly_chart(fig, use_container_width=True)

# price histogram
st.header('Price Distribution')
st.sidebar.subheader('Select Max Price')

## filters
min_price = int(df['price'].min())
max_price = int(df['price'].max())
avg_price = int(df['price'].mean())

f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)

## data selection
price_df = df.loc[df['price'] <= f_price]

## plot
fig = px.histogram(price_df, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

# other house categories
st.sidebar.title('Attributes Options')
st.title('House Attributes')

## filters to bedrooms and bathrooms
f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(df['bedrooms'].unique()))
f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(df['bathrooms'].unique()))

c1, c2 = st.beta_columns(2)
## data selection
bedrooms_df = df.loc[df['bedrooms'] <= f_bedrooms]
bathrooms_df = df.loc[df['bathrooms'] <= f_bathrooms]

## house per bedrooms
c1.header('Houses per bedrooms')
fig = px.histogram(bedrooms_df, x='bedrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

## house per bathrooms
c2.header('Houses per bathrooms')
fig = px.histogram(bathrooms_df, x='bathrooms', nbins=19)
c2.plotly_chart(fig, use_container_width=True)

## filters to floors and waterview
f_floors = st.sidebar.selectbox('Max number of floors', sorted(df['floors'].unique()))
f_waterview = st.sidebar.checkbox('Only Houses with Water View')

c1, c2 = st.beta_columns(2)
## data selection
df_floors = df.loc[df['floors'] <= f_floors]
if f_waterview:
    df_waterview = df[df['waterfront'] == 1]
else:
    df_waterview = df

## house per floors
c1.header('Houses per floors')
fig = px.histogram(df_floors, x='floors', nbins=10)
c1.plotly_chart(fig, use_container_width=True)

## house per water view
c2.header('Houses with waterfront')
fig = px.histogram(df_waterview, x='waterfront', nbins=2)
c2.plotly_chart(fig, use_container_width=True)




