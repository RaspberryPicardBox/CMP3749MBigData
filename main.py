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


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    df.show()
