#!/usr/bin/env python
# coding: utf-8

# # Recurrent Neural Network for Daily Bike-Sharing Trip Prediction
# 
# This notebook explores the potential of Recurrent Neural Networks (RNNs) for daily bike-sharing trip prediction. 
# 
# ### This notebook will focus on the following aspects:
# 
# * Data Acquisition: Explore and load a bike-sharing trip dataset and save on Hadoop filesystem.  
# * Data Preprocessing: Clean and transform the data throught pyspark for RNN analysis. 
# * Model Selection and Training: Develop and train LSTM and GRU models using TensorFlow.
# * Model Evaluation: Evaluate the performance of the trained models. 
# * Conclusion: Analyze the findings and potential benefits for bike-sharing companies.

# ### Import necessary libraries 
# 

# In[37]:


import pyspark.sql.functions as psf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, sum, lit, to_date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import keras
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.regularizers import L2


from statsmodels.tsa.seasonal import seasonal_decompose



import warnings
warnings.filterwarnings("ignore") 



# ### Apache Hadoop and PySpark for Scalable Processing
#     
#     Bike-sharing data can be vast. Traditional computing platforms might struggle to handle such large datasets efficiently. This is where Apache Hadoop and PySpark come into play.
#     
# ### Apache Hadoop: 
#     An open-source framework that facilitates distributed processing of large datasets across clusters of computers.
#     
# ### Apache Spark: 
#     Apache Spark™ is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters. PySpark is a Python-like API for data processing tasks.
# 

# In[2]:


# Create a SparkSession
spark = SparkSession.builder.appName("CA01_Sem_02").config("spark.executor.memory","10g").getOrCreate()
spark


# ### Data_Ingestion Module
#     To leverage the use of hadoop,  this module was created to read the csv files into a pyspark dataframe and save on hadoop hdfs system
#     This function will be used once, its a programaticaly way to ingest data on hadoop filesystem.

# In[3]:


# data_ingestion function: Loads all CSV files in a directory using PySpark and saves them to a specified output path on HDFS.
# Args: 
    # path: Path to the directory containing CSV files  
    # hd_path: Hadoop path on HDFS to save the processed files

def data_ingestion( path, hd_path):
    
    try:
        # Read all files in the directory matching the CSV extension (*.csv)
        df = spark.read.csv(path, header=True, inferSchema = True)

        # Save the DataFrame to Hadoop server - HDFS
        df.write.mode("overwrite").csv(hd_path, header=True)
        print('Data Ingestion Completed')

    except Exception as e:
        print("An error occurred:", e)
    
    
    


# ### Running the data_ingestion function
#     In this process, all csv files will be loaded into a spark dataframe and saved on Hadoop using the function data_ingestion created.

# In[14]:


# path to directory where the files are
path = "file:///home/hduser/Documents/CA01_Sem_02_MSc_Data_Analytics/*.csv"

# path where the files should be saved on hadoop
hd_path = "/user/hduser/bike_data/"

# Call the processing function
data_ingestion(path, hd_path)


# ### Verifying the data on hadoop
#     As the data is saved on local server, its possible to run the hdfs shell to verify if the data was savedd properly 

# In[5]:


get_ipython().system('hdfs dfs -ls /user/hduser/bike_data')


# ### Load data from hadoop
# 
#     From now, all data will be loaded from Hadoop

# In[41]:


hd_path = "/user/hduser/bike_data/"

try:
  # Read the CSV files from hadoop 
  df = spark.read.csv(hd_path, header = True, inferSchema = True)

  # Show the first 2 rows of the DataFrame
  df.show(2)
   
except Exception as e:
    print("An error occurred:", str(e))


# ### printSchema()
# The df.printSchema() function in Apache Spark is used for inspecting the schema of a DataFrame. 
# The schema essentially defines the structure of your data, specifying the names and data types of each column in the DataFrame.
# 
# **Column Name:** The name assigned to each column in the DataFrame.
# 
# **Data Type:** The specific data type of each column, such as int, string, double, etc. This indicates how the data in that column is stored and interpreted.
# 
# **Nullability:** Whether a column allows null values or not. It will be displayed as true if null values are allowed, and false if not.

# In[42]:


df.printSchema()


# ### Checking the "shape" of the dataframe
# 

# In[43]:


print((df.count(), len(df.columns)))


# ### Data Engineering 
# When it comes to using RNNs for time series forecasting or analysis, the data structure must have some level of consistency, as each row of the dataset represents one trip, A new column will be created to represent the tripa and only date and trips were selected from the dataset

# In[44]:


trips = df.select("started_at")


# In[45]:


trips.show(5)


# In[46]:


trips = trips.withColumn("started_at",to_date(col("started_at")).alias("date"))


# In[47]:


trips.show(5)


# In[48]:


trips = trips.withColumn("trips", lit(1))


# In[49]:


trips.show(5)


# In[50]:


trips.printSchema()


# # Grouping by date
# To create a consistent dataset,  all trips will be grouped by date, and sum all trips for each day.

# In[54]:


#window_spec = window(trips['started_at'], "1 day")
daily_df = trips.groupBy("started_at").agg(sum("trips").alias("total_trips"))


# In[56]:


daily_df = daily_df.orderBy("started_at")


# In[57]:


daily_df.show(10)


# In[58]:


daily_df.describe().show()


# ### Insigh from describe method
# 
# **Data Size:** 
# 
#     The count value (365) indicates that the data is for 365 days, which means a full year of daily bike trips.
#     
# **Central Tendency:** 
# 
#     The mean value (12239.27) is the average number of daily bike trips across the year. 
#     
# **Variability:**
# 
#     The stddev value (3983.05) this high standard deviation relative to the mean suggests significant variability in the number of trips per day. There could be days with much higher or lower trips compared to the average.
#     
# **Range:**
# 
#     The min value (2369) and max value (20174) shows the spread of the data, and there are days with low rides, and day with very high rides (10 times) the lower day.
# 

# ### EDA - Understanding the Data

# In[225]:


# Converting PySpark DataFrame to Pandas DataFrame
pandas_df = daily_df.toPandas()

# Saving transformed data in csv format for future usage
pandas_df.to_csv("pandas_df.csv", index = False, header = True)


# ### From naw, the data will be loaded from "pandas_df.csv"  

# In[26]:


pandas_df = pd.read_csv("pandas_df.csv")


# In[27]:


pandas_df.head()


# ### Distribution and Skewness

# In[28]:


# Calculate skewness
skewness = pd.Series(pandas_df['total_trips']).skew()

# Plot the histogram
plt.figure(figsize=(8,4))
plt.hist(pandas_df['total_trips'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Trips per Day')
plt.xlabel('Total Trips')
plt.ylabel('Frequency')
plt.grid(True)

# Add skewness as annotation
plt.annotate(f"Skewness: {skewness:.3f}", xy=(0.8, 0.8), xycoords='axes fraction', fontsize=10)
plt.show()



# ### Insight from the Distribution Plot
# 
# Most days have between 7500 and 15000 total trips. There are also a fewer number of days that have either a very low or very high number of total trips.
# 
# **Shape of the Distribution:** 
# 
#     The distribution appears to be symmetrical, but there might be a slight tail towards the higher number of trips, also the number of daily bike trips varies, but most days fall within a certain range. 
# 
# **Skewness:**
# 
#     A value of 0 indicates a symmetrical distribution, while positive values indicate a skew to the right and negative values indicate a skew to the left.
#     This negative skewness (-0.5073) suggests that the distribution has a longer tail towards lower daily trip counts compared to the average (mean). 

# ### Boxplot and Scatter Plot - Ouliers

# In[29]:


# Calculate quartiles and IQR
Q1 = pandas_df['total_trips'].quantile(0.25)
Q3 = pandas_df['total_trips'].quantile(0.75)
IQR = Q3 - Q1

lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 

# Creating a boxplot in the first subplot
ax1.boxplot(pandas_df['total_trips'])
ax1.set_title('Box Plot of Total Trips per Day')
ax1.set_xlabel('Total Trips')
ax1.set_ylabel('Value')
ax1.grid(False)

# creating a scatter plot with IQR outliers in the second subplot
ax2.scatter(pandas_df.index, pandas_df['total_trips'], color='blue', label='Inliers')
ax2.scatter(pandas_df[pandas_df['total_trips'] < lower_threshold].index, 
            pandas_df[pandas_df['total_trips'] < lower_threshold]['total_trips'], 
            color='red', marker='o', label='Outliers (Below Q1-1.5 IQR)')
ax2.scatter(pandas_df[pandas_df['total_trips'] > upper_threshold].index, 
            pandas_df[pandas_df['total_trips'] > upper_threshold]['total_trips'], 
            color='red', marker='x', label='Outliers (Above Q3+1.5 IQR)')

ax2.set_title('Scatter Plot with Distribution and IQR Outliers')
ax2.set_xlabel('Day') 
ax2.set_ylabel('Total Trips')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

plt.show()


# ### Insight from BoxPlot
# As already shown on distribution plot, the box extends from roughly 7,500 to 12,500 trips, indicating that there is variability in the number of daily trips, but most days fall within this range.
# 
# ### From Scatter plot 
# **Overall Distribution:** Most of the data points cluster around the center, what means the data is mostly in a range.
# 
# **Outliers:** Although, there are some points close to the IRQ threshold, there arent outliers.
#     
#     

# ## Analisys of Trips Per Month

# In[30]:


# Function transform_dataframe takes a DataFrame (`df`) and a window parameter (`window`)  and transforms the DataFrame based on the specified window.
# Args:  window: 'weekly' or 'monthly' and df: dataframe to be transformed, it returns the transformed dataframe

def transform_dataframe(window, df):
    if window == 'weekly':
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['weekday']=df['started_at'].dt.weekday
        df = df.groupby("weekday")["total_trips"].sum().reset_index()

        df = df.sort_values(by="weekday")
        df['weekday']=df['weekday'].map({0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'})
    else:
        df['started_at'] = pd.to_datetime(df['started_at'])
        df['month']=df['started_at'].dt.month
        df = df.sort_values(by="month")
        df = df.groupby("month")["total_trips"].sum().reset_index()
        df['month']=df['month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
       
    return df
    


# ### Applying the transform_dataframe function and plotting a weekday and month trips.

# In[31]:


weekly = transform_dataframe('weekly',pandas_df)
monthly = transform_dataframe('montly',pandas_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4)) 

ax1.bar(weekly["weekday"], weekly["total_trips"], color='skyblue', edgecolor='black')
ax1.set_xlabel('Weekday')
ax1.set_ylabel('total of Trips')
ax1.set_title('Total of Trips by Weekday')

ax2.bar(monthly["month"], monthly["total_trips"], color='skyblue', edgecolor='black')
ax2.set_xlabel('Month')
ax2.set_ylabel('Total of Trips')
ax2.set_title('Total of Trips by Month')

fig.tight_layout()

fig.show()





# ### Insight from these plots:
# 
# From the weekly plot, its possible to observe higher trips on weekdays (Monday to Friday) compared to weekends (Saturday and Sunday). This suggests that bike-sharing is likely used for commuting purposes during weekdays.
# From the Monthly plot, its possible to observe that in the Winter (Winter in Washington DC lasts from the beginning of December to March 1st and is known to be very cold and snowy), there are fewer trips.

# # Trips by day

# In[32]:


pandas_df['started_at'] = pd.to_datetime(pandas_df['started_at'])

plt.figure(figsize=(15, 5))

plt.plot(pandas_df["started_at"], pandas_df["total_trips"])

plt.xlabel("Date")
plt.ylabel("Total Trips")
plt.title("Plot of Total Trips per day")
plt.xticks(rotation=45, ha='right')
plt.grid(True) 
plt.show()


# ### Insights from the Chart:
# **Seasonality:** The chart shows a seasonality in trips, with a clear peak in the warmer months.
#  
#  **Weekly Trend:** Its possiple to observe more trips on weekdays compared to weekends. 
# 

# In[10]:


# Set index to 'started_at'
pandas_df.set_index('started_at', inplace=True)


# In[11]:


# Decompose time series (additive model)
decomposition = seasonal_decompose(pandas_df['total_trips'], model='additive')

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(pandas_df['total_trips'], label='Original')
plt.legend()

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend()

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend()

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend()

plt.tight_layout()
plt.show()


# ## Insight from this decompose
# ### Trend:
# 
# The trend chart shows an upward trend, indicating that the number of trips increases in the long term.
# 
# ### Seasonality:
# 
# As previously analyzed, there is a clear seasonal pattern in trips, with a peak in the warmer months. However, the seasonality is moderate, meaning there is still a significant amount of ridership even in colder months.
# 
# ### Residual:
# 
# The residual component displays some fluctuations around zero. These fluctuations represent random variations or other underlying patterns in ridership that are not captured by the trend and seasonality.
# 
# 
# 

# # LSTM Model Building:
# 

# For building a LSTM model, its important to observe some parameters:
#     Input layer: This layer takes the sequence of past trip counts as input. for this project, a window size of (1, 3, 5 ,7, 14, 30 ) past days was chosen to feed into the LSTM.
#     LSTM layer: This layer is the core of the model, responsible for learning long-term dependencies in the trip count data. The model will be evaluated with different layers and numbers.
#     Output layer: This layer predicts the future trip count. It will be used a single neuron with a linear activation for numerical prediction.
#        
#     TensorFlow is designed to work primarily with NumPy arrays. Because of this, pandas.core.series will be transform into NumPy arrays.

# In[12]:


# Function preprocessing: Preprocesses the bike trip data for time series forecasting.
# Args: scaler: A scaler object used to normalize the data.  look_back: The number of past time steps used to predict the next value.

def preprocessing(scaler, look_back):
        
    pandas_df = pd.read_csv("pandas_df.csv")
    trips = pandas_df.iloc[:,-1]
    
   
    # convert to numpy array
    np_trips = trips.to_numpy()
    np_trips = np_trips.astype('float32')
    np_trips = np_trips.reshape(-1,1)


    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-look_back-1):
    		a = dataset[i:(i+look_back), 0]
    		dataX.append(a)
    		dataY.append(dataset[i + look_back, 0])
    	return np.array(dataX), np.array(dataY)

    # normalize the dataset
    np_trips = scaler.fit_transform(np_trips)
    
    # split into train and test sets
    train_size = int(len(np_trips) * 0.75)
    test_size = len(np_trips) - train_size
    train, test = np_trips[0:train_size,:], np_trips[train_size:len(np_trips),:]
    
  
   
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


    
    
    trainX.shape, testX.shape
    return trainX, testX, trainY, testY, np_trips


# ### Define look_back period
# 
# After  various tests, the best period or look_back was 1 day.

# In[13]:


look_back = 1  # defining the look_back


# # 3 Models will be applied:
# * Simple LSTM
# * LSTM with Hyperparameter
# * GRU Model
# 
# 
# ### Model 01  - Simple LSTM 

# In[16]:


# create and fit the LSTM network
model1 = Sequential()
model1.add(LSTM(8, input_shape=(1, look_back)))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.summary()


# In[15]:


# scaler to normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#call the function processing
trainX, testX , trainY , testY, np_trips = preprocessing(scaler, look_back)


model1_fit = model1.fit(trainX, trainY, epochs=50 , batch_size=1, verbose=0)


# Access training history
train_history = model1_fit.history

# Get the minimum validation loss epoch 
min_val_loss_epoch = train_history['loss'].index(min(train_history['loss']))

# Print the best validation loss and corresponding epoch
print("Best Loss:", min(train_history['loss']))
print("Epoch with Best Loss:", min_val_loss_epoch + 1)  


# ### Evaluating Model - LSTM

# In[21]:


# make predictions
trainPredict = model1.predict(trainX)
testPredict = model1.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[206]:


#shift train predictions for plotting
trainPredictPlot = np.empty_like(np_trips)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = np.empty_like(np_trips)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(np_trips)-1, :] = testPredict


# In[207]:


# plot baseline and predictions

plt.figure(figsize=(15, 5))
plt.plot(scaler.inverse_transform(np_trips))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(['Number of Trips )', 'Baseline Prediction (Train)', 'Prediction (Test)'], loc='upper left')
plt.show()


# # MODEL 02   - LSTM Hyperparameter
# 

# In[208]:


model2 = Sequential()
model2.add(LSTM(128, return_sequences=True, input_shape=(1, look_back))) 
model2.add(LSTM(units=64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)))
model2.add(Dropout(0.2)) 
model2.add(LSTM(32))
model2.add(Dropout(0.2))
model2.add(Dense(16))  
model2.add(Dense(1, activation='relu'))
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.summary()


# In[209]:


# scaler to normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#call the function processing
trainX, testX , trainY , testY, np_trips = preprocessing(scaler, look_back)


#fit model2
model2_fit = model2.fit(trainX, trainY, epochs=50, validation_data=(testX,testY), batch_size=1, verbose=0)


# Access training history
train_history = model2_fit.history

# Get the minimum validation loss epoch 
min_val_loss_epoch = train_history['loss'].index(min(train_history['loss']))

# Print the best validation loss and corresponding epoch
print("Best Loss:", min(train_history['loss']))
print("Best Validation Loss:", min(train_history['val_loss']))

print("Epoch with Best Loss and Validation Loss:", min_val_loss_epoch + 1)  


# ### Evaluating Model 02 - LSTM with hyperparameter

# In[210]:


# make predictions
trainPredict = model2.predict(trainX)
testPredict = model2.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[211]:


# shift train trainPredictPlot for plotting
trainPredictPlot = np.empty_like(np_trips)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test testPredictPlot for plotting
testPredictPlot = np.empty_like(np_trips)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(np_trips)-1, :] = testPredict


# In[212]:


# plot baseline and predictions
plt.figure(figsize=(15, 5))

plt.plot(scaler.inverse_transform(np_trips))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# # Model 03 - GRU
# 

# In[38]:


model3 = keras.Sequential()
model3.add(GRU(128, return_sequences=True, input_shape=(1, look_back)))  
model3.add(GRU(units=64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.01)))
model3.add(Dropout(0.2))  
model3.add(GRU(32))
model3.add(Dropout(0.2))
model3.add(Dense(16)) 
model3.add(Dense(1))
model3.compile(loss='mean_squared_logarithmic_error', optimizer='adam')
model3.summary()


# In[39]:


# scaler to normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

#call the function processing
trainX, testX , trainY , testY, np_trips = preprocessing(scaler, look_back)


#fit model3
model3_fit = model3.fit(trainX, trainY, epochs=50, validation_data=(testX,testY), batch_size=1, verbose=0)


# Access training history
train_history = model3_fit.history

# Get the minimum validation loss epoch 
min_val_loss_epoch = train_history['loss'].index(min(train_history['loss']))

# Print the best validation loss and corresponding epoch
print("Best Loss:", min(train_history['loss']))
print("Best Validation Loss:", min(train_history['val_loss']))

print("Epoch with Best Loss and Validation Loss:", min_val_loss_epoch + 1)  


# ### Evaluating Model - GRU

# In[40]:


# make predictions
trainPredict = model3.predict(trainX)
testPredict = model3.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[41]:


# shift train trainPredictPlot for plotting
trainPredictPlot = np.empty_like(np_trips)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict


# shift test testPredictPlot for plotting
testPredictPlot = np.empty_like(np_trips)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(np_trips)-1, :] = testPredict


# In[42]:


# plot baseline and predictions
plt.figure(figsize=(15, 5))

plt.plot(scaler.inverse_transform(np_trips))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# # Result
# 
# All models achieved lower training RMSE compared to the set RMSE test.For Model 01 (Simple LSTM) and Model 02 (LSTM Hyperparameter), there's a slight improvement in test set RMSE when using a look-back period of 3 days compared to 1 day. 
# 
# However, when we increase the look-back to 7 days, performance is not improved much. This maybe indicates that these models benefit from taking in a few days of historical information, but can't effectively capture long-term dependencies. 
# 
# Model 03 (GRU) maintains roughly the testing set RMSE across all look-back periods, suggesting that GRU architectures may be less sensitive to specific look-back period than LSTM ones. 
# 
# All the same, the test set RMSE keeps persisting at a high level across all kinds of Model 3, implying limitations in prediction accuracy. The high RMSE scores across all models make clear that daily bike ridership probably depends on complex factors besides the historical trip data used for training. Here are a few possible reasons: These models were built using only historical trip data. Adding in more features such as weather information (temperature, precipitation), holidays or events unique to certain days, and day of the week effects, might improve our predictions somewhat.
# 
