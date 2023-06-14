from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession
from hdfs import InsecureClient
from uploadToExploitationZone import getIdealistaRDD


hdfs_cli = InsecureClient("http://10.4.41.46:9870", user="bdm")
subdir = '/user/bdm/formatted_zone/idealista'
files = [f"{subdir}/{name}" for name in hdfs_cli.list('/user/bdm/formatted_zone/idealista')]

def startSpark():
    """
    start spark session
    """
    spark = SparkSession.builder \
        .appName("Read Parquet Files") \
        .getOrCreate()
    spark.conf.set("spark.sql.parser.escapedStringLiterals", "true")
    return spark

spark = startSpark()
# Create a SparkContext
sc = SparkContext(appName="LinearRegressionExample")

# Define the data RDD
dataRDD = sc.parallelize([
    ((2020, 'El Carmel'), ((149000, 92175668), 687.025)),
    ((2021, 'Can Baró'), ((130000, 84215678), 645.324)),
    ((2022, 'Gràcia'), ((180000, 79134567), 723.543))
])

# Prepare the data for linear regression
labeled_data = dataRDD.map(lambda x: LabeledPoint(x[1][1], Vectors.dense(x[1][0])))

# Split the data into training and testing sets
training_data, testing_data = labeled_data.randomSplit([0.8, 0.2])

# Train the linear regression model
model = LinearRegressionWithSGD.train(training_data)

# Make predictions on the testing set
predictions = model.predict(testing_data.map(lambda x: x.features))

# Zip the predicted values with the actual labels
results = predictions.zip(testing_data.map(lambda x: x.label))

# Print the predicted and actual values
print("Predicted\tActual")
for pred, actual in results.collect():
    print(f"{pred:.2f}\t\t{actual:.2f}")

# Stop the SparkContext
sc.stop()