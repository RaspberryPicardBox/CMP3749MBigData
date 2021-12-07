def openSparkDirectory():
    direc = open("directories.txt", "r")
    ret = direc.read()
    direc.close()
    return ret


import findspark
findspark.init(openSparkDirectory())
import pyspark
from pyspark.sql import SparkSession
import matplotlib as plt
import numpy as np


def repairData(df):
    print("Cleaning df of count: {0}".format(df.count()))
    df = df.dropDuplicates()
    df = df.dropna()
    print("Done cleaning df - new count: {0}".format(df.count()))
    return df


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    df = repairData(df)


