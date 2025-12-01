# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/market-basket-analysis. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to prepare the data with which we will build a market basket recommender.  

# COMMAND ----------

# MAGIC %md ## Introduction 
# MAGIC
# MAGIC Market basket analysis, a variation of associative (affinity) analysis, examines products frequently purchased together.  The rules produced by such analysis capture how the selection of one or more items may indicate preferences for additional items and would seem to provide a natural basis for the creation of recommendations:
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/mba_recommender.png' width=400>
# MAGIC
# MAGIC In this series of notebooks, we will examine first how these rules are generated and then how they may be employed to make recommendations based on items in a shopper's cart.

# COMMAND ----------

# MAGIC %md The basic building block of our market basket recommender is transactional data identifying the products customers purchased together at checkout. The popular [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis) provides us a nice collection of such data with over 3.3 million grocery orders placed by over 200,000 Instacart users over a nearly 2-year period across of portfolio of nearly 50,000 products.
# MAGIC
# MAGIC **NOTE** Due to the terms and conditions by which these data are made available, anyone interested in recreating this work will need to accept the terms and rules before downloading the data files from Kaggle and uploading them to a folder structure as described below.
# MAGIC
# MAGIC The primary data files available for download are organized as follows. You can save the data permanently under a pre-defined [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) that named */mnt/instacart*:
# MAGIC We have automated this data preparation step for you and used a */tmp/instacart_market_basket* storage path throughout this accelerator.
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>
# MAGIC
# MAGIC Let's download the data now:

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import window as w

# COMMAND ----------

# MAGIC %md ## Step 1: Load the Data
# MAGIC
# MAGIC The basic building block of our market basket recommender is transactional data identifying the products customers purchased together at checkout. The popular Instacart dataset provides us a nice collection of such data with over 3.3 million grocery orders placed by over 200,000 Instacart users over a nearly 2-year period across of portfolio of nearly 50,000 products. (You can read more about this dataset [here](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2).)
# MAGIC
# MAGIC The setup of the data files making up this dataset is addressed in the *MB 01* notebook. In this notebook, we will read these data and convert them to Delta tables providing us with more efficient access to the data throughout the remainder of the notebooks.  The database schema associated with these tables is as follows:
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>

# COMMAND ----------

# DBTITLE 1,Get Configuration Values
# MAGIC %run "./01_Configuration"

# COMMAND ----------

# DBTITLE 1,Create Database
_ = spark.sql('DROP DATABASE IF EXISTS instacart CASCADE')
_ = spark.sql('CREATE DATABASE instacart')

# COMMAND ----------

# MAGIC %md The orders data is pre-divided into *prior* and *training* evaluation sets, where the *training* dataset represents the last order placed in the overall sequence of orders associated with a given customer.  The *prior* dataset represents those orders that proceed the *training* order.  While not typical of most market basket analyses, we will generate our rules on the *prior* subset, leveraging the *training* subset to evaluate our recommendations.
# MAGIC
# MAGIC It is important to note that we are calculating a *days_prior_to_last_order* field on the orders data. This field is used in other recommenders that we have built on this same dataset but will not be used here.  It is being included in our data prep to keep our data consistent between solutions:

# COMMAND ----------

# DBTITLE 1,Orders
# define schema for incoming data
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

# calculate days until final purchase (not needed by this solution but included for consistency with some previously developed assets that use this data)
win = (
  w.Window.partitionBy('user_id').orderBy(f.col('order_number').desc())
  )

# read data from csv
_ = (
  spark
    .read
      .csv(
        config['orders_files'],
        header=True,
        schema=orders_schema
        )
    .withColumn(  # (not needed by this solution but included for consistency with some previously developed assets that use this data)
        'days_prior_to_last_order', 
        f.sum('days_since_prior_order').over(win) - f.coalesce(f.col('days_since_prior_order'),f.lit(0))
        ) 
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('instacart.orders')
  )

# present the data for review
display(
  spark
    .table('instacart.orders')
    .orderBy('user_id','order_number')
  )

# COMMAND ----------

# MAGIC %md The remaining assets are fairly straightforward and can be loaded to Delta tables as follows:

# COMMAND ----------

# DBTITLE 1,Products
# define schema for incoming data
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

# read data from csv
_ = (
  spark
    .read
      .csv(
        config['products_files'],
        header=True,
        schema=products_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('instacart.products')
  )

# present the data for review
display(
  spark.table('instacart.products')
  )

# COMMAND ----------

# DBTITLE 1,Order Products
# define schema for incoming data
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

# read data from csv
_ = (
  spark
    .read
      .csv(
        config['order_products_files'],
        header=True,
        schema=order_products_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('instacart.order_products')
  )

# present the data for review
display(
  spark.table('instacart.order_products')
  )

# COMMAND ----------

# DBTITLE 1,Departments
# define schema for incoming data
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

# read data from csv
_ = (
  spark
    .read
      .csv(
        config['departments_files'],
        header=True,
        schema=departments_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('instacart.departments')
  )

# present the data for review
display(
  spark.table('instacart.departments')
  )

# COMMAND ----------

# DBTITLE 1,Aisles
# define schema for incoming data
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

# read data from csv
_ = (
  spark
    .read
      .csv(
        config['aisles_files'],
        header=True,
        schema=aisles_schema
        )
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('instacart.aisles')
  )

# present the data for review
display(
  spark.table('instacart.aisles')
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Explore the Data
# MAGIC
# MAGIC With our data loaded, we might take a moment to perform a little exploratory analysis to inform decisions we will make in subsequent notebooks. As mentioned above, our data consists of over 3.3 million purchases recorded across a portfolio of nearly 50,000 products.  In the *prior* subset on which we will build our rules, the transaction count drops to 3.2 million with nearly all available products represented in that dataset: 

# COMMAND ----------

# DBTITLE 1,Orders & Products in Prior Period
# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   COUNT(DISTINCT y.order_id) as orders,
# MAGIC   COUNT(DISTINCT x.product_id) as available_products, 
# MAGIC   COUNT(DISTINCT y.product_id) as purchased_products
# MAGIC FROM instacart.products x
# MAGIC LEFT OUTER JOIN (  -- purchases in prior period
# MAGIC   SELECT
# MAGIC     a.order_id,
# MAGIC     b.product_id
# MAGIC   FROM instacart.orders a
# MAGIC   INNER JOIN instacart.order_products b
# MAGIC     ON a.order_id=b.order_id
# MAGIC   WHERE 
# MAGIC     a.eval_set='prior'
# MAGIC   ) y
# MAGIC   ON x.product_id=y.product_id

# COMMAND ----------

# MAGIC %md With each order, customers averaged about 10 unique products though there is considerable variability and a long, right-hand skew to this number. The larger the basket size, the more product combinations we will need to explore as we generate rules but with so much variation in basket size, it is very difficult for us to estimate the number of combinations we might encounter. Our worst case is that with 50,000 products we might need to examine roughly (3^50000 - 2^50001 + 1) rules, a [staggeringly large number](https://www.wolframalpha.com/input/?i=extrema+calculator&assumption=%7B%22F%22%2C+%22GlobalExtremaCalculator%22%2C+%22curvefunction%22%7D+-%3E%223%5E50000+-+2%5E50001+%2B+1%22) calculated using the formula presented [here](https://www-users.cse.umn.edu/~kumar001/dmbook/ch6.pdf).  Given associations between products, we expect the actual number of actual combinations in the data to be much lower though still large enough to cause concerns regarding computational efficiency:

# COMMAND ----------

# DBTITLE 1,Products in Orders (Prior Period)
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   MIN(products_in_order) as products_in_order_min,
# MAGIC   MAX(products_in_order) as products_in_order_max,
# MAGIC   AVG(products_in_order) as products_in_order_avg,
# MAGIC   STDDEV(products_in_order) as products_in_order_stdev
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     COUNT(DISTINCT product_id) as products_in_order
# MAGIC   FROM instacart.order_products a
# MAGIC   INNER JOIN instacart.orders b
# MAGIC     ON a.order_id=b.order_id
# MAGIC   WHERE b.eval_set='prior'
# MAGIC   GROUP BY a.order_id
# MAGIC   )

# COMMAND ----------

# DBTITLE 1,Products in Orders (Prior Period)
# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   COUNT(DISTINCT product_id) as products_in_order
# MAGIC FROM instacart.order_products a
# MAGIC INNER JOIN instacart.orders b
# MAGIC   ON a.order_id=b.order_id
# MAGIC WHERE b.eval_set='prior'
# MAGIC GROUP BY a.order_id

# COMMAND ----------

# MAGIC %md The potential for computational challenges has lead to the generation of numerous algorithms intended to efficiently examine a dataset for *interesting* associations. In the next notebook, we'll touch on this a bit more, but here we might examine some data management strategies that can help reduce the number of combinations we must explore.
# MAGIC
# MAGIC First, we might consider eliminating products we do not intend to recommend or use as the basis for a recommendation. A commonly cited example in grocery scenarios are bags purchased to carry items from the store.  While technically part of a shopper's cart, we don't expect there to be any interesting associations with specific products.  The trigger for the purchase of a bag is typically the number of items being purchased and whether or not a customer brought bags of their own to the store.
# MAGIC
# MAGIC In addition, we might consider dropping seasonal items which are either not available or will likely not sell with the same magnitude as found in the historical data. Pumpkin spice flavored items are an excellent example. In North American markets, these items sell very well in the Fall but tend to sell extremely poorly (if they are even available) in warmer months.
# MAGIC
# MAGIC Finally, we might consider building our rules around items aggregated at a level above the SKU. This might help us build better product associations between items where minor differences in packaging might dilute the strength of rules generated from our data. For example, if single-serve bags of potato chips sell well with fountain drinks, we might not expect there to be much value in differentiating between 20-oz, 32-oz and 44-oz fountain drinks for the purposes of making a recommendation. However, we should be careful to not make too gross a generalization as we might loose interesting, niche associations in the data. We must also keep in mind the nature of the recommendation we intend to make.  If the inclusion of single-serve potato chips into a cart is intended to trigger a recommendation of a fountain drink, we must either recommend a specific SKU or be prepared to recommend fountain drinks more broadly and have a mechanism by which the customer eventually arrives at a specific SKU.

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.