# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/market-basket-analysis. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to generate association rules with which we can make product recommendations. 

# COMMAND ----------

# DBTITLE 1,Get Configuration Values
# MAGIC %run "./01_Configuration"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.fpm import FPGrowth
import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %md ##Step 1: Assemble Item Sets
# MAGIC
# MAGIC With our data loaded, we might now start the process of examining product associations. Referred to as an *item set*, the collection of products purchased together within a given transaction provides us the basis for exploring product associations. While transactions containing just one product don't provide us with product combinations to explore, they need to be included in this input set so that the frequency with which individual products occur across all transactions can be properly calculated.
# MAGIC
# MAGIC For readability purposes, we will build our item sets using friendly product names.  Later, we will revisit this, building our item sets using less memory-demanding integer-based product IDs:

# COMMAND ----------

# DBTITLE 1,Construct Item Sets
item_sets = (
  spark
    .table('instacart.order_products')
    .join( spark.table('instacart.orders'), on='order_id')
    .filter(f.expr("eval_set='prior'")) # limit to prior subset
    .join(  # join to products to get product names
      spark.table('instacart.products'), 
      on='product_id'
      )
    .groupBy('order_id')
      .agg(f.collect_set('product_name').alias('items')) # collect_set automatically removes duplicate elements from the array
  )

display(item_sets.limit(5)) # limit display to conserve space on screen

# COMMAND ----------

# MAGIC %md ## Step 2: Explore Associations in the Dataset
# MAGIC
# MAGIC Given the large number of product combinations found in a dataset, most algorithms require products to meet certain minimum levels of *support* in order to be considered for rule generation. Support refers to the frequency with which a product or product combination occurs within the overall dataset. The *a priori* rule (frequently spelled *Apriori*) reasons that if a product or product combination occurs with low frequency, *i.e.* has low support, product combinations containing it must be found with the same or lower support. Using this rule, many algorithms (including the [distributed frequent pattern growth algorithm](https://ieeexplore.ieee.org/document/8054308) implemented within Spark) define a minimum support threshold for products and product combinations and exclude from consideration those rules and descendent rules that do not meet the threshold.
# MAGIC
# MAGIC With this in mind, let's take a look at the support associated with individual products in the dataset. This might help us to determine an appropriate value for minimum support:

# COMMAND ----------

# DBTITLE 1,Calculate Support for Each Individual Product
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   b.product_name,
# MAGIC   COUNT(DISTINCT a.order_id) as orders,
# MAGIC   COUNT(DISTINCT a.order_id) / (SELECT COUNT(DISTINCT order_id) FROM instacart.order_products) as support
# MAGIC FROM instacart.order_products a
# MAGIC INNER JOIN instacart.products b
# MAGIC   ON a.product_id=b.product_id
# MAGIC GROUP BY b.product_name
# MAGIC ORDER BY support DESC

# COMMAND ----------

# MAGIC %md The support for individual products are pretty low.  With so many products in the portfolio and a large number of transactions, this is not surprising. The default minimum support level of 0.3 associated with our implementation of FPGrowth is clearly too high for this dataset, but what should we set the threshold to?
# MAGIC
# MAGIC A common practice is to specify a minimum number of occurrences based on knowledge of the business scenario and to use that to calculate the minimum support threshold. As our goal is to make recommendations, the inclusion of more data in the analysis will hurt us from a computational standpoint but will better ensure we have something to recommend as customers engage. We might therefore set a relatively low bar of 100 occurrences of a product or product combination and see where this leads us.
# MAGIC
# MAGIC Because we expect the number of rules explored by the algorithm to be quite large, it is helpful for us to run this on a fairly large cluster and to aggressively distributed the data across the cluster workers by configuring the *numPartitions* argument appropriately:

# COMMAND ----------

# DBTITLE 1,Generate Rules
# components of minimum support threshold
num_of_transactions = item_sets.count() 
min_transactions = 100

# configure model
fpGrowth = FPGrowth(
              itemsCol='items', 
              minSupport= min_transactions/num_of_transactions, 
              minConfidence=0.0, 
              numPartitions=sc.defaultParallelism * 100
              )

# fit model to item set data
model = fpGrowth.fit(item_sets)

# count number of rules generated
model.associationRules.count()

# COMMAND ----------

# MAGIC %md At a 100-occurrences minimum, we produce nearly 1.7 million product associations.  We can see those associations here:

# COMMAND ----------

# DBTITLE 1,Display Rules
display(
  model.associationRules
  )

# COMMAND ----------

# MAGIC %md The association rules generated by the model are presented in the form of an antecedent and a consequent.  In market basket analysis terminology, the antecedent represents the set of products that indicate the consequent product.  The *confidence* value associated with each rule indicates the frequency with which the consequent occurs whenever the antecedent is present. A value of 0 would indicate the two never occur together in the dataset while a value of 1 would indicate the two always occur together in the data. 
# MAGIC
# MAGIC You may have noticed in the model configuration that we had the option to specify a minimum confidence level by which we could eliminate rules.  As our goal is to generate recommendations, we might leave low confidence rules in place to ensure we have a larger number of options for presenting products in response to customer actions.  In other applications of market basket analysis, low confidence rules would typically be seen as of limited value and might be eliminated.
# MAGIC
# MAGIC A lift metric is also provided with each rule. *Lift* tells us something about the degree to which the antecedent and consequent occur together relative to what would be expected by random chance given their overall frequencies in the dataset. At a lift value of 1.0, the antecedent and consequent are believed to co-occur by chance.  Above 1.0, the objects are being placed in a shopper's cart at higher frequencies than would be expected by random chance. And the higher that value goes, the stronger the relationship between the two.
# MAGIC
# MAGIC The problem with lift as it is calculated here is that it is very sensitive to the frequency with which a product is found in the overall dataset. Low frequency items may receive very high lift values simply because the few times those items showed up in a cart, they happen to have been in there in combination with some other product.  And for this reason, a [*normalized lift*](https://www.scss.tcd.ie/disciplines/statistics/tech-reports/07-01.pdf) value is often calculated when there is the possibility of low support items being used in a market basket analysis.
# MAGIC
# MAGIC To calculate a normalized lift, we need to determine the frequency by which the antecedent and consequent occurs within the dataset.  We can retrieve that information from the *freqItemsets* attribute associated with our model:

# COMMAND ----------

# DBTITLE 1,Retrieve Item Sets Represented within the Association Rules
# persist frequent item sets to disk
(
  model
    .freqItemsets
    .withColumn('items', f.array_sort('items')) # sort items for easier comparison later
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save( config['root_path'] + '/tmp/freq_item_sets' )
  )

# retrieve frequent items sets to dataframe
freq_item_sets = spark.table('DELTA.`{0}`'.format(config['root_path'] + '/tmp/freq_item_sets'))

# display items in set
display(freq_item_sets)

# COMMAND ----------

# MAGIC %md You'll notice in the last cell that we are persisting the frequent item sets associated with our model to temporary storage.  In our next step, we will be performing a complex join between these data and the association rules and we have found that it performs more consistently if we first persist this and the association rules to disk.  With that in mind, let's tackle the temporary persistence of the association rules:

# COMMAND ----------

# DBTITLE 1,Temporarily Persist Association Rules
# persist association rules to disk
(
  model
    .associationRules
    .withColumn('antecedent', f.array_sort('antecedent')) # sort antecedent for easier comparisons later
    .repartition(sc.defaultParallelism * 10)
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema', 'true')
      .save( config['root_path'] + '/tmp/assoc_rules')
  )

# retrieve association rules to dataframe
assoc_rules = spark.table('DELTA.`{0}`'.format(config['root_path'] + '/tmp/assoc_rules'))

# COMMAND ----------

# MAGIC %md Now we can join our rules to frequency data in the frequent item sets table. We are doing this by comparing array fields for equality.  Because arrays are ordered within Spark, it's important to note we sorted the arrays prior to their persistence in the previous cells.  This will ensure that a simple equality comparison between arrays will work as expected:

# COMMAND ----------

# DBTITLE 1,Retrieve Antecedent & Consequent Frequencies
rules_with_frequencies = (
  assoc_rules.alias('rule')
    .join(
      freq_item_sets.alias('ant'),
      on=f.expr('rule.antecedent = ant.items')  # arrays are sorted above to allow for this comparison
      )
    .join(
      (
        freq_item_sets.alias('con')    # get set of frequent items that are just one product
        .filter(f.expr('size(con.items)=1'))
      ),
      on=f.expr('rule.consequent=con.items')
     )
  .selectExpr(
    'antecedent',
    'consequent',
    'support',
    'confidence',
    'lift',
    'ant.freq as ant_freq',
    'con.freq as con_freq'
  )
)

display(rules_with_frequencies)

# COMMAND ----------

# MAGIC %md Once all the required data is joined, we can calculate the normalized lift by calculating the range of potential lift values we could observe for a given rule and then normalizing the score within this range. The logic behind this calculation gets back to the notion that as there are more and more occurrences of the products in the antecedent and the products in the consequent, they will inevitably have to co-occur within transactions. This is used to establish a floor for the lift value.  At the ceiling, the lesser of the frequency of either the items in the antecedent or the items in the consequent will define an upper bound for lift. Within this range, the lift value is normalized to reside between 0 and 1 with higher values indicating stronger affinities between the antecedent and consequent:

# COMMAND ----------

# DBTITLE 1,Calculate Normalized Lift
# get count of all orders
num_of_transactions = item_sets.count()

# calculate normalized lift
norm_assoc_rules = (
  rules_with_frequencies
    .withColumn('lift_min', f.expr('({0} * greatest(0, ant_freq + con_freq - {0})) / (ant_freq * con_freq)'.format(num_of_transactions))) # 
    .withColumn('lift_max', f.expr('({0} * least(ant_freq, con_freq)) / (ant_freq * con_freq)'.format(num_of_transactions)))
    .withColumn('lift_norm', f.expr('(lift - lift_min) / (lift_max - lift_min)'))
    .select(
      'antecedent',
      'consequent',
      'support',
      'confidence',
      'lift',
      'lift_norm'
      )
  )

display(
  norm_assoc_rules
  )

# COMMAND ----------

# MAGIC %md There are [numerous other measures](https://mhahsler.github.io/arules/docs/measures) with which we might assess the quality of the rules we have generated. For our purposes, normalized lift should be sufficient but other metrics may need to be derived depending on a specific business scenario.

# COMMAND ----------

# MAGIC %md ## Step 3: Assemble Rules for Recommendations
# MAGIC
# MAGIC Having explored how we might go about generating rules with a more transparent but memory-consuming collection of product name item sets, we'll now repeat our work using the more frequently employed integer IDs for our products:

# COMMAND ----------

# DBTITLE 1,Generate Item Set
item_sets = (
  spark
    .table('instacart.order_products')
    .join( spark.table('instacart.orders'), on='order_id')
    .filter(f.expr("eval_set='prior'")) # limit to prior subset
    .groupBy('order_id')
      .agg(f.collect_set('product_id').alias('items')) # collect_set automatically removes duplicate elements from the array
  )

display(item_sets.limit(5)) # limit display to conserve space on screen

# COMMAND ----------

# DBTITLE 1,Generate Rules
# components of minimum support threshold
num_of_transactions = item_sets.count() 
min_transactions = 100

# configure model
fpGrowth = FPGrowth(
              itemsCol='items', 
              minSupport= min_transactions/num_of_transactions, 
              minConfidence=0.0, 
              numPartitions=sc.defaultParallelism * 100
              )

# fit model to item set data
model = fpGrowth.fit(item_sets)

# COMMAND ----------

# DBTITLE 1,Temporarily Persist Data for Normalized Lift Calculations
# persist frequent item sets to disk
(
  model
    .freqItemsets
    .withColumn('items', f.array_sort('items')) # sort items for easier comparison later
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save( config['root_path'] + '/tmp/freq_item_sets' )
  )

# retrieve frequent items sets to dataframe
freq_item_sets = spark.table('DELTA.`{0}`'.format(config['root_path'] + '/tmp/freq_item_sets'))

# persist association rules to disk
(
  model
    .associationRules
    .withColumn('antecedent', f.array_sort('antecedent')) # sort antecedent for easier comparisons later
    .repartition(sc.defaultParallelism * 10)
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema', 'true')
      .save( config['root_path'] + '/tmp/assoc_rules')
  )

# retrieve association rules to dataframe
assoc_rules = spark.table('DELTA.`{0}`'.format(config['root_path'] + '/tmp/assoc_rules'))

# COMMAND ----------

# DBTITLE 1,Calculate Normalized Lift
# get count of all orders
num_of_transactions = item_sets.count()

# calculate normalized lift
norm_assoc_rules = (
  assoc_rules.alias('rule')
    .join(
      freq_item_sets.alias('ant'),
      on=f.expr('rule.antecedent = ant.items')  # arrays are sorted above for comparison
      )
    .join(
      (
        freq_item_sets.alias('con')    # get set of frequent items that are just one product
        .filter(f.expr('size(con.items)=1'))
      ),
      on=f.expr('rule.consequent=con.items')
     )
    .withColumn('lift_min', f.expr('({0} * greatest(0, ant.freq + con.freq - {0})) / (ant.freq * con.freq)'.format(num_of_transactions))) # 
    .withColumn('lift_max', f.expr('({0} * least(ant.freq, con.freq)) / (ant.freq * con.freq)'.format(num_of_transactions)))
    .withColumn('lift_norm', f.expr('(rule.lift - lift_min) / (lift_max - lift_min)'))
    .select(
      'antecedent',
      'consequent',
      'support',
      'confidence',
      'lift',
      'lift_norm'
      )
  )

display(
  norm_assoc_rules
  )

# COMMAND ----------

# DBTITLE 1,Persist Rules with Scores for Use in Recommendations
_ = (
  norm_assoc_rules
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('instacart.assoc_rules')
  )

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.