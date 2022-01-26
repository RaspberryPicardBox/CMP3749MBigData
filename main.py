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
from pyspark.sql.functions import lit, when, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns


def repairData(df):  # Repair and return the data by removing duplicates and Null values
    print("Cleaning df of count: {0}".format(df.count()))
    df = df.dropDuplicates()
    df = df.dropna()
    print("Done cleaning df - new count: {0}".format(df.count()))
    return df


def summary(df):  # Show summary of the data
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


def correlation(df):  # Show correlation matrix of df
    df = df.drop("Status")
    df = df.toPandas()
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    plt.show()


def split(df):  # Return split data for train and test
    vectorAssembler = VectorAssembler(inputCols=df.drop('Status').columns, outputCol='features')
    df = vectorAssembler.transform(df).drop('Power_range_sensor_1', 'Power_range_sensor_2', 'Power_range_sensor_3',
                                            'Power_range_sensor_4', 'Pressure_sensor_1', 'Pressure_sensor_2',
                                            'Pressure_sensor_3', 'Pressure_sensor_4', 'Vibration_sensor_1',
                                            'Vibration_sensor_2', 'Vibration_sensor_3', 'Vibration_sensor_4')

    stringIndexer = StringIndexer(inputCol="Status", outputCol="labelIndex")
    df = stringIndexer.fit(df).transform(df).drop('Status')

    (train, test) = df.randomSplit([0.7, 0.3])

    return train, test


def decTree(train, test):  # Show decision tree
    dt = DecisionTreeClassifier(labelCol='labelIndex', featuresCol='features')
    model = dt.fit(train)
    predictions = model.transform(test)
    predictions.show()

    evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print("Test error = {}".format(1.0-accuracy))
    print("Accuracy = {}".format(accuracy))


def linearSupportVector(train, test):
    lsvc = LinearSVC(maxIter=50, regParam=0.0, labelCol='labelIndex', featuresCol='features')
    model = lsvc.fit(train)
    predictions = model.transform(test)
    predictions.show()

    evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    print("Test error = {}".format(1.0 - accuracy))
    print("Accuracy = {}".format(accuracy))


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()

    df = spark.read.csv("nuclear_plants_small_dataset.csv", inferSchema=True, header=True)  # Load data from csv file
    df = repairData(df)  # Repair the data

    """print("-------Summary where Status is Normal-------\n")
    summary(df.where(df["Status"] == "Normal"))  # Summarise the data where Status is Normal

    print("-------Summary where Status is Abnormal-------\n")
    summary(df.where(df["Status"] == "Abnormal"))  # Summarise the data where Status is Abnormal

    print("-------Correlation matrix of DF-------\n")
    correlation(df)  # Shows correlation matrix of df"""

    print("-------Shuffling and splitting data...-------\n")
    train, test = split(df)
    """print("-------Train Set-------\nLength: {}\n".format(train.count()))
    train.show()
    print("-------Test Set-------\nLength: {}\n".format(test.count()))
    test.show()"""

    # decTree(train, test)
    linearSupportVector(train, test)
