\documentclass{article}
\usepackage{hyperref}
\usepackage{enumitem}

\title{Final Project Plan}
\author{Raymond Blaha and Paul Hwang}
\date{\today}

\begin{document}

\maketitle

\section{Introduction / Question}
Can we predict wildfire risk zones using historical wildfire, vegetation, and climate data in the California region?

\section{Data Collection}

Since I am collecting data from two different souces one challenge will being to combine them into one cohesive dataset. 
Current the order of events in which data collection will be occuring: 

\begin{enumerate}
    \item 1.0 MODIS: I created an Earthdata account to get access to the most updated data. 
    \item 1.1 Explored the NASA Earthdata Search. Where I filtered the MODIS14A. 
    \item 1.2 Used the chmod 777 download.txt in local terminal to start the download process.
    \item 1.3 Used login information in local terminal to download the data.
    \item 2.0 USGS: Went in an downloaded the data. 
    \item 2.1 Merged the MODIS and USGS data via geographical location.
    \item 2.2 Download the climate data.
    \item 2.3 Extract the temporal climate and perform merge with the already joined MODIS and GAP data.
    \item 2.4 Split the data into training, vaildation, and testing.
    \item 3.0 Formulate the model to have a seemless execution and flow from each model to the next. 
    \item 3.1 Load the three models to test the model on unseen data. 
    \item 3.2 Visualize the results.
    \item 3.3 Compare the results to the labeled ground truth data with fire labels. 
\end{enumerate}

\section{Data Exploration}

\subsection{MODIS}

After downloading the satellite data which took 10 hours. I inspected the data and the metadata of the HD5 files to get a better understanding of how we could merge the data. 


\subsection{USGS}
After downloading the data, the most important and relevant data is the $\textit{raster\_values}$ and the spatial data. Since the vegetation data does not really change over time.
Thebest approach to merge was to convert the USGS data from WGS 84 referencing the spatial instances of vegetatation to latitude and longitiude. For every instance or non instance of the MODIS we can combine
the two datasets. This allows us to keep the temporal isntances of fires and non fires, while having vegitiation data. Merging took approx 30 hours to complete. For the sake of computational efficiency, 
we extracted the 2D spatial data from the USGS data. So it still can be read into the CNN while keeping the temporal instances of the MODIS data.

\subsection{Climate Data}
Since the data is temporal. We can merge the data by the temporal instances, while trying to keep in mind location. 

\section{Data Preprocessing}
We will need to prepocess certain columsn in which we want to input into the models. First instances will be some of the metadata from MODIS about fires, time, the raster values from USGS, and the climate data.


\section{Modeling}
Regarding the autoencoder, I will first have the autoencoder find features in the satellite data. Once that is done, 
I will then have the CNN analyze the features the 2D identify spatial patterns that will be key for predicting wildfire risk zones.
Finally, the LSTM will give us prospecitve on the temporal climate instances which will give the model more context to predict wildfire risk zones.

\section{Visualization of Results}
Results will be presented in a simple format, such as risk zone maps based on the model's analysis.

Another aspect that I would like to explore is comparing the model's predictions to the ground truth data. To see how accurate the model is and or if there are any inefficiencies in the model.
Then make the proper adjustments to the model if needed. 

\end{document}
