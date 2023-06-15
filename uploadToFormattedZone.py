import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, input_file_name
from pyspark.sql.types import DateType, StringType
from pyspark.sql.functions import col
from hdfs import InsecureClient
import os

# Set the HDFS server address
hdfs_server = "hdfs://10.4.41.46:27000"
file_size_bytes = 128 * 1024 * 1024
hdfs_path = "/user/bdm/persistent_landing"
hdfs_cli = InsecureClient("http://10.4.41.46:9870", user="bdm")
subdirs = [f"{hdfs_path}/{name}" for name in hdfs_cli.list(hdfs_path) if hdfs_cli.status(f"{hdfs_path}/{name}")['type'] == 'DIRECTORY']
subdir = '/user/bdm/persistent_landing/idealista'
files = [f"{subdir}/{name}" for name in hdfs_cli.list('/user/bdm/persistent_landing/idealista')]
target_hdfs_path = "/user/bdm/formatted_zone"
def startSpark():
    """
    start spark session
    """
    spark = SparkSession.builder \
        .appName("Read Parquet Files") \
        .getOrCreate()
    spark.conf.set("spark.sql.parser.escapedStringLiterals", "true")
    return spark

def readIdealistaFiles(sprak, files):
    """
    looping over all idealista parquet files, checking logical key exists and removing duplications, adding date of the file to the data
    :param sprak:
    :param files:
    :return: finalRDD
    """
    flag = True
    for file in files:
        extract_filename_udf = udf(lambda filepath: os.path.basename(filepath)[:10], StringType())
        convert_to_date_udf = udf(lambda date_str: datetime.datetime.strptime(date_str, "%Y_%m_%d").date(),
                                  DateType())  # to convert into a date
        df = sprak.read.parquet(hdfs_server + file)
        # Not taking records with columns without the logical key
        if 'neighborhood' not in df.columns:
            continue
        # Add date column, filter out null values for logical key and convert to rdd
        rdd = df.withColumn("date", convert_to_date_udf(extract_filename_udf(input_file_name())))\
            .filter(col('neighborhood').isNotNull())\
            .rdd
        if flag:
            fullRDD = rdd.cache()
            flag = False
        else:
            fullRDD = fullRDD.union(rdd).cache()
    uniqueRDD = fullRDD\
        .map(lambda row:(row['propertyCode'], row['date']))\
        .reduceByKey(lambda date1, date2: max(date1, date2))\
        .map(lambda row: ((row[0],row[1]),(row[1])))\
        .cache()
    finalRDD = fullRDD\
    .map(lambda row: ((row['propertyCode'], row['date']), (row))) \
    .join(uniqueRDD)
    return finalRDD

def readParquetFiles(spark, folder_name):
    """
    read parquet file from hdfs /user/bdm/persistent_landing from all folders but idealista in the path and turn it into RDD
    :param spark:
    :param folder_name:
    :return: rdd
    """
    parquet_files = spark.read.parquet(hdfs_server + hdfs_path + '/' + folder_name)
    rdd = parquet_files.rdd
    return rdd

def reconIncoming(incRDD, lkpRDD):
    """
    reconcile income data with the lookup tables
    :param incRDD:
    :param lkpRDD:
    :return:
    """
    mapRDD = incRDD.map(lambda row: ((row.Nom_Districte, row['Nom_Barri']), row))
    reconRDD = mapRDD\
        .join(lkpRDD) \
        .map(lambda row: ((row[1][1].district_n_reconciled, row[1][1].neighborhood_n_reconciled, row[1][0]['Any']), row))
    return reconRDD

def reconRent(rentRDD, lkpRDD):
    """
    reconcile rent data with the lookup tables
    :param rentRDD:
    :param lkpRDD:
    :return: reconRDD
    """
    mapRDD = rentRDD.map(lambda row: ((row.Nom_Districte, row['Nom_Barri']), row))
    reconRDD = mapRDD\
        .join(lkpRDD)\
        .map(lambda row: ((row[1][1].district_n_reconciled,row[1][1].neighborhood_n_reconciled,row[1][0]['Any']), row[1][0]))
    return reconRDD

def reconIdealista(ideaRDD, lkpRDD):
    """
    reconciled Idealista data with the lookup tables
    :param ideaRDD:
    :param lkpRDD:
    :return: reconRDD
    """
    mapRDD = ideaRDD.map(lambda row: ((row[1][0].neighborhood),row))
    reconRDD = mapRDD\
        .join(lkpRDD)\
         .map(lambda row: ((row[1][1][5]), row[1][0]))
    return reconRDD

def delete_hdfs_directory(path):
    """
    delete formatted zone if the folder exists
    :param path:
    """
    if not hdfs_cli.status(path, strict=False):
        print(f"The folder {path} does not exist.")
        return
    for root, dirs, files in hdfs_cli.walk(path):
        for file in files:
            hdfs_cli.delete(os.path.join(root, file))
    hdfs_cli.delete(path, recursive=True)
    print(f"The folder {path} was deleted.")

def getSchemaIdealista(RDD):
    """
    align idealista schema to the data
    :param RDD:
    :return: df
    """
    newRDD = RDD.map(lambda x: (x[0],x[1][1][0].district ,x[1][0][0],x[1][0][1], x[1][1][0].price,
    x[1][1][0].size,x[1][1][0].rooms,x[1][1][0].bathrooms,x[1][1][0].latitude,x[1][1][0].longitude,
    x[1][1][0].operation,x[1][1][0].propertyType,x[1][1][0].floor,x[1][1][0].status,x[1][1][0].hasLift)).cache()

    df = newRDD.toDF()
    column_names = ["neighborhood", "district", "propertyCode", "date", "price", "size", "rooms", "bathrooms",
                    "latitude", "longitude", "operation", "propertyType", "floor", "status", "hasLift"]
    # Rename columns and assign data types
    for i, column_name in enumerate(column_names):
        df = df.withColumnRenamed("_" + str(i+1), column_name)
    return df

def getSchemaIncome(RDD):
    """
    align income schema to the data
    :param RDD:
    :return: df
    """
    newRDD = RDD.map(lambda x: (x[0][0], x[0][1], x[0][2], x[1][1][0].Població, x[1][1][0].Index_RFD_Barcelona_100))
    df = newRDD.toDF()
    column_names = ["district", "neighborhood", "year", "populatonBCN", "Index_RFD_Barcelona_100"]
    # Rename columns and assign data types
    for i, column_name in enumerate(column_names):
        df = df.withColumnRenamed("_" + str(i + 1), column_name)
    return df

def getSchemaRent(RDD):
    """
    align rent schema to the data
    :param RDD:
    :return: df
    """
    newRDD = RDD.map(lambda x: (x[0][0], x[0][1], x[0][2], x[1].Trimestre , x[1].Lloguer_mitja, x[1].Preu))
    df = newRDD.toDF()
    column_names = ["district", "neighborhood", "year", "Trimestre", "Lloguer_mitja", "Preu"]
    # Rename columns and assign data types
    for i, column_name in enumerate(column_names):
        df = df.withColumnRenamed("_" + str(i + 1), column_name)
    return df

def toHDFS(df, path, name):
    """
    insert the relevant source of rdd data into hdfs to /user/bdm/formatted_zone/{name}
    :param df:
    :param path:
    :param name:
    """
    records_per_file = int(file_size_bytes / df.rdd.getNumPartitions())
    repartitioned_df = df.coalesce(int(file_size_bytes / records_per_file))
    repartitioned_df.write \
        .mode("overwrite") \
        .parquet(hdfs_server + path + f"/{name}")
    print(f"Data for {name} was created on HDFS successfully")

if __name__ == "__main__":
    spark = startSpark()
    idealistaRDD = readIdealistaFiles(spark,files).cache()
    incomeRDD = readParquetFiles(spark,'opendatabcn-income/').cache()
    rentRDD = readParquetFiles(spark,'opendata-rent/').cache()
    lkpRDD = readParquetFiles(spark, 'lookup_tables/').cache()
    mapLkpRDD = lkpRDD\
        .map(lambda row: ((row.district, row.neighborhood), row))\
        .sortByKey()\
        .distinct()\
        .cache()
    testLkpRDD = lkpRDD \
        .map(lambda row: ((row.neighborhood), row)) \
        .sortByKey() \
        .distinct() \
        .cache()

    # Rconciliation
    incReconRDD = reconIncoming(incomeRDD, mapLkpRDD)
    rentReconRDD = reconRent(rentRDD,mapLkpRDD)
    idealistaReconRDD = reconIdealista(idealistaRDD, testLkpRDD).cache()

    idealista_df = getSchemaIdealista(idealistaReconRDD)
    income_df = getSchemaIncome(incReconRDD)
    rent_df = getSchemaRent(rentReconRDD)
    delete_hdfs_directory(target_hdfs_path)

    toHDFS(idealista_df, target_hdfs_path,"idealista")
    toHDFS(income_df,target_hdfs_path,"income")
    toHDFS(rent_df,target_hdfs_path,"rent")

    spark.stop()