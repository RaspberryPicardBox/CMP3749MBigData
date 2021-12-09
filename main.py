def openSparkDirectory():
    direc = open("directories.txt", "r")
    ret = direc.read()
    direc.close()
    return ret


import findspark

findspark.init(openSparkDirectory())
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as pf
import matplotlib as plt
import numpy as np


def repairData(df):
    print("Cleaning df of count: {0}".format(df.count()))
    df = df.dropDuplicates()
    df = df.dropna()
    print("Done cleaning df - new count: {0}".format(df.count()))
    return df


def summary(df):
    df = df.drop('Status')

    for header in df.columns:
        col = df.select(df[header])

        minimum = col.agg({header: "max"})
        maximum = col.agg({header: "min"})
        mean = col.agg({header: "mean"})
        median = col.agg(pf.expr('percentile_approx({0}, 0.5)'.format(header)).alias("median"))
        mode = df.groupBy(header).count().orderBy("count", ascending=False)
        mode = mode.select(mode["count"].alias("mode"))
        deviation = col.agg({header: "stddev"})

        print("-------------------- {0} --------------------".format(header))

        minimum.join(maximum).join(mean).join(median).join(deviation).join(mode).show(1)


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)
    # df = repairData(df)

    summary(df.where(df["Status"] == "Normal"))
