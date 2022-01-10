def openSparkDirectory():  # Finding PySpark on local machine
    direc = open("directories.txt", "r")
    ret = direc.read()
    direc.close()
    return ret


import findspark
findspark.init(openSparkDirectory())

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as pf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def repairData(df):  # Repair the data by removing duplicates and Null values
    print("Cleaning df of count: {0}".format(df.count()))
    df = df.dropDuplicates()
    df = df.dropna()
    print("Done cleaning df - new count: {0}".format(df.count()))
    return df


def summary(df):  # Summary of the data
    df = df.drop("Status")

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

        minimum.join(maximum).join(mean).join(median).join(deviation).join(mode).show(1)  # Only show top 1 result

        df2 = col.toPandas()  # Make a temporary copy of the column in toPandas format
        df2.boxplot()  # Make a boxplot

        fig = plt.gcf()
        fig.canvas.manager.set_window_title(col)  # Rename the window title
        plt.show()

        # TODO: Show box plot for each group


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)  # Load data from csv file
    #df = repairData(df)  # Repair the data

    print("Summary where Status is Normal")
    summary(df.where(df["Status"] == "Normal"))  # Summarise the data where Status is Normal

    print("Summary where Status is Abnormal")
    summary(df.where(df["Status"] == "Abnormal"))  # Summarise the data where Status is Abnormal

    # TODO: Show correlation matrix for each feature
    # TODO: Decision tree
