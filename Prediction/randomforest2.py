from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import csv
from datetime import *
from dateutil.parser import parse

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
import shutil, sys

import numpy as np
import math

sc = SparkContext()
sqlContext = SQLContext(sc)

MAXYEAR = 2016
MINYEAR = 2004

def ParseDate(t):
    try:
        dt = parse(t, fuzzy=True).date()
    except ValueError as e:
        dt = date.today()
    return dt

# read the file from loacal or uplaod to hdfs
crimeFile = sc.textFile("file:///home/ubuntu/crime/raw_data/row06-15.csv")
#crimeFile = sc.textFile("file:///home/ubuntu/crime/raw_data/row16.csv")

crimeHeader = crimeFile.filter(lambda l: "CMPLNT_NUM" in l)
crimeNoHeader = crimeFile.subtract(crimeHeader)
crimes = crimeNoHeader.mapPartitions(lambda x: csv.reader(x, delimiter=",")).filter(lambda row: len(row) > 15 and row[1] is not None).map(lambda row: (int('0'+row[14]), ParseDate(row[1])))  
crimes = crimes.filter(lambda row: row[1].year>MINYEAR and row[1].year<MAXYEAR and row[0]>0 )

# List of all the PCTs
PCTs = crimes.map(lambda row: (row[0])).distinct().zipWithIndex()

# biuld the index of PCTs
PCTsDict = dict(PCTs.collect())
print PCTs.first()

# ( (precient, data), number 0f crime)
crimeCounts = crimes.map(lambda row: ((PCTsDict[row[0]], row[1]), 1)).reduceByKey(lambda x,y: x + y)
print crimeCounts.top(2)

# Obtain all dates in the dataset
Dates = crimeCounts.map(lambda row: (row[0][1])).distinct()

# Generate all possible PCT-year-week combinations from 2006 to 2015
allPCTDates = PCTs.values().cartesian(Dates)

# RDD ((PCT, date), countCrime)
missingPCTDates = allPCTDates.subtract(crimeCounts.keys()).distinct()
allCrimeCounts = crimeCounts.union(missingPCTDates.map(lambda row: (row, 0)))

# Process the temperature
# Load the historical temperature for the city and filter it for the years 2005 to 2015
temperature = sc.textFile("file:///home/ubuntu/crime/raw_data/NYNEWYOR.txt").map(lambda line: [float(i) for i in line.split()]).filter(lambda row: row[2]>MINYEAR and row[2]<MAXYEAR).map(lambda row: (date(int(row[2]), int(row[0]), int(row[1])), row[3]))

# joinedData RDD:(countCrime,(weekday, PCT, avg_temperature))
joinedData = allCrimeCounts.map(lambda row: ((row[0][1]), (row[0][0], row[1]))).join(temperature).map(lambda row: ((row[0].weekday(), row[1][0][0], row[1][1]), row[1][0][1])).reduceByKey(lambda x,y: x + y).map(lambda row: LabeledPoint(row[1], [row[0][0], row[0][1], row[0][2]]))
print joinedData.top(2)

# Split the crime counts into training and test datasets
(training, test) = joinedData.randomSplit((0.9, 0.1))

# Train a Random Forest model to predict crimes
model = RandomForest.trainRegressor(training, categoricalFeaturesInfo = { 0: 7, 1: len(PCTsDict)},
                                    numTrees = 7, featureSubsetStrategy = "auto",
                                    impurity='variance', maxDepth=10, maxBins = len(PCTsDict))

#### Predicting crimes for a day####
PCTsDictInverse = dict((v, k) for k, v in PCTsDict.items())

data = []
for weekday in range(7):
    for tempForecast in range(10,100,5):
        # Test dataset for each beat with next week's info
        predictday = sc.parallelize(tuple([(weekday, PCT ,tempForecast) for PCT in range(len(PCTsDict))]))
        predictionsday = model.predict(predictday).zip(predictday.map(lambda row: PCTsDictInverse[row[1]])).sortByKey(False)
        # Obtain the top 10 beats with highest likelihood of crime
        topCrimePCTs = predictionsday.take(10)
        #topCrimePCT = [x[0] for x in topCrimePCTs]
        row = [weekday, tempForecast]
        row.extend(topCrimePCTs)
        data.append(row)

with open('output.csv', 'wb') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(data)
    outfile.close()

# Save and load model

output_dir = 'file:///home/ubuntu/crime/myRandomForestClassificationModel2'
shutil.rmtree(output_dir, ignore_errors=True)
model.save(sc, output_dir)

