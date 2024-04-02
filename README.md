# Final Project Plan

**Authors:** Raymond Blaha and Paul Hwang  
**Date:** [11/28/2023]

## Introduction / Question

Can we predict wildfire risk zones using historical wildfire, vegetation, and climate data in the California region?

## Data Collection

Since I am collecting data from two different sources, one challenge will be to combine them into one cohesive dataset. Currently, the order of events in which data collection will be occurring:

1. **MODIS**: I created an Earthdata account to get access to the most updated data.
   - Explored the NASA Earthdata Search where I filtered the MODIS14A.
   - Used `chmod 777 download.txt` in local terminal to start the download process.
   - Used login information in the local terminal to download the data.
2. **USGS**: Went in and downloaded the data.
   - Merged the MODIS and USGS data via geographical location.
   - Downloaded the climate data.
   - Extracted the temporal climate and performed merge with the already joined MODIS and GAP data.
   - Split the data into training, validation, and testing.
3. **Model Formulation**: Formulate the model to have a seamless execution and flow from one model to the next.
   - Loaded the three models to test the model on unseen data.
   - Visualized the results.
   - Compared the results to the labeled ground truth data with fire labels.

## Data Exploration

### MODIS

After downloading the satellite data, which took 10 hours, I inspected the data and the metadata of the HD5 files to get a better understanding of how we could merge the data.

### USGS

After downloading the data, the most important and relevant data is the *raster_values* and the spatial data. Since the vegetation data does not really change over time, the best approach to merge was to convert the USGS data from WGS 84, referencing the spatial instances of vegetation to latitude and longitude. For every instance or non-instance of the MODIS, we can combine the two datasets. This allows us to keep the temporal instances of fires and non-fires while having vegetation data. Merging took approx 30 hours to complete. For the sake of computational efficiency, we extracted the 2D spatial data from the USGS data. So it still can be read into the CNN while keeping the temporal instances of the MODIS data.

### Climate Data

Since the data is temporal, we can merge the data by the temporal instances, while trying to keep in mind location.

## Data Preprocessing

We will need to preprocess certain columns in which we want to input into the models. The first instances will be some of the metadata from MODIS about fires, time, the raster values from USGS, and the climate data.

## Modeling

Regarding the autoencoder, I will first have the autoencoder find features in the satellite data. Once that is done, I will then have the CNN analyze the features to identify spatial patterns that will be key for predicting wildfire risk zones. Finally, the LSTM will give us perspective on the temporal climate instances, which will give the model more context to predict wildfire risk zones.

## Visualization of Results

Results will be presented in a simple format, such as risk zone maps based on the model's analysis.

Another aspect that I would like to explore is comparing the model's predictions to the ground truth data. To see how accurate the model is and or if there are any inefficiencies in the model. Then make the proper adjustments to the model if needed.
