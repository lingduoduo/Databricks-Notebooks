# Databricks notebook source
# MAGIC %md ## Cold Start

# COMMAND ----------

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# COMMAND ----------

### Users without any historical action clicks in last 7 days

# -- This case cold-start recommendations by scoring and ranking items based on time-decayed clicks across all users, highlighting trending or popular items even when no personal history exists for the given client.
# -- No clicks on  user history in the last 7 days: It aggregates click data from all users, making it ideal for recommending items to new or low-activity clients.

# -- Recency-aware ranking: By applying the decay factor 1 / (1 + delta), it emphasizes recent interactions, surfacing currently relevant items.

# -- Popularity-based recommendation: Items are ranked by their decayed click scores, effectively prioritizing trending content likely to appeal broadly.  
                

# COMMAND ----------

recommended_actions = spark.sql(f"""WITH clicks AS (
    SELECT
        client_id,
        click_object_id AS item_id,
        click_details_caption AS title,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'") AS unix_timestamp,
        COUNT(*) AS clicks
    FROM
        onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
    WHERE
        click_object_id IS NOT NULL 
        AND action = 'actions'
    GROUP BY
        client_id,
        click_object_id,
        click_details_caption,
        TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
),

clicks_with_max AS (
    SELECT
        *,
        MAX(unix_timestamp) OVER () AS max_timestamp
    FROM clicks
)

SELECT
    client_id,
    item_id,
    title,
    SUM((1.0 / (1 + ((max_timestamp - unix_timestamp) / (24 * 60 * 60 * 100)))) * clicks) AS weighted_clicks
FROM
    clicks_with_max
GROUP BY
    client_id,
    item_id,
    title
ORDER BY
    client_id,
    weighted_clicks DESC;
""")

# COMMAND ----------

display(recommended_actions)

# COMMAND ----------

import boto3

boto3_session = boto3.Session(
    botocore_session=dbutils.credentials.getServiceCredentialsProvider(
        'service-cred-nas-lifion_ml-sdq-dit'
    )
)
s3_client = boto3_session.client('s3') 

# COMMAND ----------

import pandas as pd
from datetime import datetime

bucket_name = "ml-models-bucket-appbuild-02"
ts = datetime.now()

file = f"cold_start_{ts}.csv"
recommended_actions.toPandas().to_csv(file, index=False)

# Upload file to S3
file_path = f"recommended-actions/{file}"
response = s3_client.put_object(Bucket=bucket_name, Body=open(file, "rb"), Key=file_path)
status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
if status == 200:
    print(f"Successful S3 put_object response. Status - {status}")

# COMMAND ----------

response = s3_client.get_object(Bucket=bucket_name, Key=file_path)

# Display the DataFrame
# display(df)
status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
if status == 200:
    print(f"Successful S3 get_object response. Status - {status}")
    df = pd.read_csv(response.get("Body"))

# COMMAND ----------

# MAGIC %md ## Step 1: Collect Data

# COMMAND ----------

df_clicks = spark.sql(f"""
SELECT
    _token_associate_id AS user_id,
    click_object_id AS item_id,
    TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'") AS timestamp,
    SUM(click) AS rating
FROM
    onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
WHERE
    click_object_id IS NOT NULL AND action = "actions"
GROUP BY 1, 2, 3
""")

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
# df_clicks = spark.sql(f"""
# SELECT
#     _token_associate_id AS user_id,
#     click_object_id AS item_id,
#     client_id AS categories,
#     TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'") AS timestamp,
#     SUM(click) AS rating
# FROM
#     onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
# WHERE
#     click_object_id IS NOT NULL AND action = "actions"
# GROUP BY 1, 2, 3, 4
# -- UNION
# -- SELECT
# --     'unknown' AS user_id,
# --     click_object_id AS item_id,
# --     client_id AS categories,
# --     max(TO_UNIX_TIMESTAMP(time_stamp, "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'")) AS timestamp,
# --     0 AS rating
# -- FROM
# --     onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
# -- WHERE
# --     click_object_id IS NOT NULL AND action = "actions"
# -- GROUP BY 2, 3
""")

# COMMAND ----------

display(df_clicks)

# COMMAND ----------

# DBTITLE 1,Retrieve Data
print(f'Total:\t{df_clicks.count()}')

# COMMAND ----------

import os
import pandas as pd

def get_ave_score(df):
    """
    Args:
        input_path_file: user rating file
    Returns:
        dict: key = item_id, value = average score (rounded to 3 decimals)
    """
    # Group by item_id and calculate the mean rating
    ave_score_series = df.groupby("item_id")["rating"].sum().round(3)

    # Convert Series to dict
    return ave_score_series.to_dict()

# COMMAND ----------

# 'b3cc3ceac4d24c2e843aa13078bd2f8e'
pdf = df_clicks.toPandas()
ave_score = get_ave_score(pdf)
display(ave_score)

# COMMAND ----------

def get_latest_timestamp(df):
    """
    Args:
        input_path_file: user rating file (columns: userid, item_id, rating, timestamp)
    """
    # Drop rows with missing timestamp, convert to int
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    # Get latest timestamp
    latest = df["timestamp"].max()
    return latest

# COMMAND ----------

print(get_latest_timestamp(pdf))

# COMMAND ----------

def topk_normalized(group, topk=10):
    """
    Rank and normalize top-k categories for a user.
    Args:
        group (DataFrame): Rows corresponding to a single user's category scores.
        topk (int): Number of top categories to return.
    Returns:
        list of tuples: [(category, normalized_score), ...]
    """
    top = group.sort_values("weighted_score", ascending=False).head(topk)
    total = top["weighted_score"].sum()
    if total == 0:
        return list(zip(top["cate"], [0] * len(top)))
    return list(zip(top["cate"], (top["weighted_score"] / total).round(3)))

# COMMAND ----------

def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Returns:
        time score
    """
    df = df_clicks.toPandas()
    fix_time_stamp = get_latest_timestamp(df)
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/100
    return round(1/(1+delta), 3)

# COMMAND ----------

    # -- size(collect_set(client_id)) AS num_categories
df_items = spark.sql(f"""
SELECT
    click_object_id AS item_id,
    click_details_caption AS title,
    concat_ws('|', collect_set(client_id)) AS categories
FROM
    onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
WHERE
    click_object_id IS NOT NULL and action = "actions"
GROUP BY
    click_object_id,
    click_details_caption
 """)

# df_items = spark.sql(f"""
# SELECT
#     click_object_id AS item_id,
#     click_details_caption AS title
# FROM
#     onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
# WHERE
#     click_object_id IS NOT NULL and action = "actions"
# GROUP BY
#     click_object_id,
#     click_details_caption
#  """)

# COMMAND ----------

display(df_items)

# COMMAND ----------

# MAGIC %md ## Step 2: Define the Model
# MAGIC

# COMMAND ----------

ave_score['ff3b3f8391b7490c95f3c0c693327aa5']

# COMMAND ----------

def get_item_cate(ave_score, df, topk=10):
    """
    Args:
        ave_score: dict, key = item_id, value = average rating score
        input_path_file: item info file (with categories)
    Returns:
        item_cate: dict, key = item_id, value = {category: ratio}
        cate_item_sort: dict, key = category, value = top item_ids list
    """
    # Read file skipping header

    # Drop rows with missing values in categories
    df = df.dropna(subset=["categories"])

    # Split categories into lists
    df["cate_list"] = df["categories"].apply(lambda x: x.split("|"))

    # Calculate equal ratio for each category per item
    df["cate_ratio"] = df["cate_list"].apply(lambda x: round(1 / len(x), 3) if len(x) > 0 else 0)

    # Build item_cate: item_id -> {cate: ratio}
    item_cate = {}
    for _, row in df.iterrows():
        item_id = str(row["item_id"])
        ratio = row["cate_ratio"]
        item_cate[item_id] = {cate: ratio for cate in row["cate_list"]}

    # Build category -> {item_id: score}
    df["score"] = df["item_id"].apply(lambda x: ave_score.get(x, 0))
    cate_item_map = {}

    for _, row in df.iterrows():
        for cate in row["cate_list"]:
            if cate not in cate_item_map:
                cate_item_map[cate] = []
            cate_item_map[cate].append((row["item_id"], row["score"]))

    # For each category, sort items by score and keep topk
    cate_item_sort = {
        cate: [item_id for item_id, _ in sorted(items, key=lambda x: x[1], reverse=True)[:topk]]
        for cate, items in cate_item_map.items()
    }

    # print(item_cate)
    # print(cate_item_sort)

    return item_cate, cate_item_sort

# COMMAND ----------

df = df_items.toPandas()
item_cate, cate_item_sort = get_item_cate(ave_score, df, topk=100)

# COMMAND ----------

item_cate

# COMMAND ----------

cate_item_sort['002']

# COMMAND ----------

def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Returns:
        time score
    """
    df = df_clicks.toPandas()
    fix_time_stamp = get_latest_timestamp(df)
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/100
    return round(1/(1+delta), 3)

# COMMAND ----------

def topk_normalized(group, topk=2):
    """
    Rank and normalize top-k categories for a user.
    Args:
        group (DataFrame): Rows corresponding to a single user"s category scores.
        topk (int): Number of top categories to return.
    Returns:
        list of tuples: [(category, normalized_score), ...]
    """
    top = group.sort_values("weighted_score", ascending=False).head(topk)
    total = top["weighted_score"].sum()
    if total == 0:
        return list(zip(top["cate"], [0] * len(top)))
    return list(zip(top["cate"], (top["weighted_score"] / total).round(3)))

# COMMAND ----------

def get_up(item_cate, df, topk=10):
    """
    Compute user preferences from ratings and item-category mappings.

    Args:
        item_cate (dict): {item_id: {cate: ratio}}
        input_path_file (str): Path to user rating CSV file.
        topk (int): Number of top categories to return per user.

    Returns:
        dict: {userid: [(cate1, ratio1), (cate2, ratio2), ...]}
    """
    # Load and filter ratings
    df = df[df["rating"] >= 1]
    df = df[df["item_id"].isin(item_cate)]
    # Apply time score
    df["time_score"] = df["timestamp"].apply(get_time_score)

    # Expand item-cate mapping into rows
    expanded_rows = []
    for _, row in df.iterrows():
        item_id = row["item_id"]
        user_id = row["user_id"]
        rating = row["rating"]
        time_score = row["time_score"]
        for cate, ratio in item_cate[item_id].items():
            weighted = rating * time_score * ratio
            expanded_rows.append((user_id, cate, weighted))
    expanded_df = pd.DataFrame(expanded_rows, columns=["user_id", "cate", "weighted_score"])

    # Aggregate by user-category
    agg_df = expanded_df.groupby(["user_id", "cate"])["weighted_score"].sum().reset_index()

    return {
        user_id: topk_normalized(group.drop(columns="user_id"))
        for user_id, group in agg_df.groupby("user_id")
    }

# COMMAND ----------

df = df_clicks.toPandas()
res = get_up(item_cate, df, topk=100)

# COMMAND ----------

res['5f28ba10-cc11-4e62-8667-9e57b5778ab3']

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Train an Initial Model
# define the als model
als = ALS(
  rank=100,
  maxIter=5,  
  userCol='user_id', 
  itemCol='item_id', 
  ratingCol='rating',
  implicitPrefs=True
  )

# train the model
model = als.fit(ratings_sampled_train)

# COMMAND ----------

# MAGIC %md Using our model, we can get predictions for our testing set as follows:

# COMMAND ----------

# DBTITLE 1,Make Predictions
predictions = model.transform(ratings_sampled_test)

display(predictions)

# COMMAND ----------

# MAGIC %md You may notice in the predictions above that there are several user-product combinations that return a prediction of *NaN*.  Our model has been trained on a subset of users and products and not every user or product in the test set made it into the training set.  Because of this, we don't have latent factors for those items which makes it impossible for us to generate predictions for user-product combinations involving them.  By default, the model returns these predictions as *NaN* values.
# MAGIC
# MAGIC You can change this behavior by setting the model's **coldStartStrategy** setting to *drop*.  This will cause those values to be dropped from the returned set of predictions:

# COMMAND ----------

# DBTITLE 1,Make Predictions (Dropping Unknown Users & Products)
model.setColdStartStrategy('drop')

predictions = model.transform(ratings_sampled_test)

display(predictions)

# COMMAND ----------

# MAGIC %md So, how should we evaluate our model? Remember that we are working with implicit ratings.  In the [original paper](https://ieeexplore.ieee.org/document/4781121) on which the ALS algorithm is based, the authors spend quite a bit of time explaining how implicit ratings aren't actual ratings but instead are indicators of a binary preference, *i.e.*  the user prefers this product (1) or does not (0).  Our implicit ratings are better understood as measures of confidence that a user has a preference for a product.  That confidence can be used to suggest how confident we might be in our own recommendation of a product to a user where we would place the items we are most confident in recommending higher up in the list of our recommendations.
# MAGIC
# MAGIC This might feel like we are playing a bit with language, but this concept affects how we go about calculating the values that form our predictions. When we set the *implicitPrefs* argument to *True*, we told the model to be less concerned with getting predicted ratings close to the provided ratings value and instead to focus on getting predicted ratings in an appropriate order relative to each other.
# MAGIC
# MAGIC Because of this, we can't really evaluate this model based on predicted values relative to provided ratings (as would typically be done with a mean squared error or similar metric) but instead in terms of the sequence of items recommended in terms of the sequence of items actually selected by the user.  In other words, we need a ranking evaluator.
# MAGIC
# MAGIC The most commonly employed ranking evaluator is *MAP@K*.  MAP@K stands for *mean average precision @ k* and it takes a little work to develop an intuition around the metric.  If you want to skip that part, just know that MAP@K improves as we move from 0.0 closer to 1.0.
# MAGIC
# MAGIC Okay, so how do we interpret MAP@K?  Consider a list of *k* number of items we might recommend to a user.  These items are sequenced from most likely to be preferred to least likely.  The *k* number of items reflects the number of items we actually intend to present the user though most evaluators set this value to something like 5 to 10 to focus on how well we get the preferred items at the very top of any list of recommendations.
# MAGIC
# MAGIC Looking over this list of *k* items, we start with the first position and ask if this item was a selected item.  If it is, it gets a *precision* of 1 at k=1.  If it's not, it gets a precision of 0.  We then look at the first and second positions and ask how many of the presented items were selected.  If all were selected, we again get a *precision* score of 1 at k=2.  If none were selected, we get a precision score of 0 at k=2.  And if only one was selected, we get a precision score of 1/2 or 0.5, again at k=2. We continue looking at the first through third, then the first through fourth and so on and so on until we've calculated *precision* scores for each first through *k*th position. 
# MAGIC
# MAGIC We then average those precision scores to get the *average precision* for the recommendations for this particular user. If we repeat this calculation of *average precision* for each user, can then average the average precision scores across all users to arrive at a *mean average precision* score across the dataset. That's our *mean average precision @ k* metric.
# MAGIC
# MAGIC The challenge with MAP@K as an evaluation metric is that it sets an incredibly high bar for selections.  It also is focused on items we've actually selected in the past and in some ways is penalizing us for suggesting new products.  The trick in working with MAP@K is to accept that you're likely to produce lower scores for most recommenders.  Our goal isn't necessarily to push MAP@K to 1.0, but instead to use the metric to compare different recommenders for their relative performance.  In other words, don't evaluate a recommender as good or bad based on its MAP@K score.  Consider its value in driving a higher MAP@K score relative to your next best recommender option.
# MAGIC
# MAGIC To calculate MAP@K for our recommender, we need to decide a value for *k*.  We might choose 10 as that seems like a reasonably sized list of items to present and below that position (depending on our application) we might expect the user to enter into more of a browsing mode of item engagement that depends less on recommendation strength.  We can then ask our model to recommend the top 10 items for each user as follows:

# COMMAND ----------

# DBTITLE 1,Get Top 10 Recommendations per User
display(
  model.recommendForAllUsers(10)
  )


# COMMAND ----------

# MAGIC %md It's a little frustrating that the ALS model doesn't return the recommendations in the format required by our ranking evaluator.  So, we'll need to strip out just the *product_id* values from the array of recommendations while preserving the sequence of those recommendations.  We can do that by first exploding our recommendations in a manner that generates a column capturing the position of the resulting value in our original array.  From there, we will rebuild our list of products in the right sequence using a windowed version of the *collect_list* function.  The *order by* clause in that window definition will cause a list of one value to be generated for our first item, and list of two values for our second item, and so on and so on.  For that reason, we'll get the largest of our lists for each using a *max* aggregation:

# COMMAND ----------

# DBTITLE 1,Get Top 10 Recommendations per User (Just Product IDs)
display(
  model
    .recommendForAllUsers(10)
    .select( 
      'user_id',
      fn.posexplode('recommendations').alias('pos', 'rec') 
      )
    .withColumn('recs', fn.expr("collect_list(rec.item_id) over(partition by user_id order by pos)"))
    .groupBy('user_id')
      .agg( fn.max('recs').alias('recs')) 
  )

# COMMAND ----------

# MAGIC %md Now we get our actuals:

# COMMAND ----------

# MAGIC %md Now we can combine our actuals and predicted selections to perform the MAP@K evaluation:
# MAGIC
# MAGIC **NOTE** Even though our item column values are integers and will typically be so, the ranking evaluator expects these values to be delivered as double-precision floating point values.  We've added a cast statement to each dataset definition to tackle this.

# COMMAND ----------

# DBTITLE 1,Calculate Map @ 10
k = 10

predicted = (
  model
    .recommendForAllUsers(k)
    .select( 
      'user_id',
      fn.posexplode('recommendations').alias('pos', 'rec') 
      )
    .withColumn('recs', fn.expr("collect_list(rec.item_id) over(partition by user_id order by pos)"))
    .groupBy('user_id')
      .agg( fn.max('recs').alias('recs'))
    .withColumn('prediction', fn.col('recs').cast('array<double>')) # cast the data to the types expected by the evaluator
  )

actuals = (
  ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(item_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>')) # cast the data to the types expected by the evaluator
  )

# evaluate the predictions
eval = RankingEvaluator( 
  predictionCol='prediction',
  labelCol='label',
  metricName='precisionAtK',
  k=k
  )

eval.evaluate( predicted.join(actuals, on='user_id') )

# COMMAND ----------

# MAGIC %md The MAP@K value above is quite low when we consider perfect MAP@K is 1.0.  That said, our goal is not to push towards a perfect score but instead to use MAP@K to compare the performance of models relative to each other.   

# COMMAND ----------

# MAGIC %md ##Step 3: Tune Hyperparamters
# MAGIC
# MAGIC We now have all the elements in place to start tuning our model.  With that in mind, let's look at some of the model parameters we previously ignored.  The critical ones in terms of prediction quality are as follows:
# MAGIC </p>
# MAGIC
# MAGIC * **maxIter** - the number of cycles between user and item optimizations to employ in training the model. The more cycles we give, the better the predictions but the longer the training time.
# MAGIC * **rank** - the number of latent factors to calculate for each of the user and item submatrices
# MAGIC * **regParam** - the regularization parameter controlling the gradient decent algorithm used during latent factor optimization. Should this be  greater than 0.0 and as high as 1.0.  There's an interesting discussion on this parameter in [this white paper](https://doi.org/10.1007/978-3-540-68880-8_32).
# MAGIC * **alpha** - the parameter multiplied against our implicit ratings in order to expand the influence of high scores.  This factor is often 1 or (much) higher.
# MAGIC * **nonnegative** - allow predicted values to go negative.  If you are using the predictions as simply a ranking mechanism (as we are doing here), leave this at its default value of False.
# MAGIC </p>
# MAGIC
# MAGIC **NOTE** The higher the **rank** parameter (as well as the **maxIter** parameter up to a point), the better your model should perform (up to a point) but the longer it will take to process.  Instead of performing hyperparameter tuning on *rank* (which should almost always gravitate to the highest value you will allow within a given set of time constraints), consider testing processing time duration and model performance and make a decision about the highest value you want to fix for that parameter. (Same goes for *maxIter*.)
# MAGIC
# MAGIC Other parameters associated with the model affect training performance.  These include:
# MAGIC </p>
# MAGIC
# MAGIC * **blockSize** - This controls the preferred block size of the model.  More details on this can be reviewed [here](https://issues.apache.org/jira/browse/SPARK-20443), but you'll typically leave this value alone.
# MAGIC * **numItemBlocks** and **numUserBlocks** - the number of blocks to employ as part of the distributed computation of either the item or user matrices.  The default value is 10. You might play with these values to see how they affect performance with matrices of different sizes and complexity but we'll leave these alone.
# MAGIC </p>
# MAGIC
# MAGIC
# MAGIC With these parameters in mind, let's define a hyperparameter search space that we can use with an intelligent search conducted by [hyperopt](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/index.html).  To learn more about the definition of hyperopt search spaces, please refer to [this document](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).  We'd also suggest you take a look at [this excellent blog post](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) on how to get the most out of hyperopt:

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
  'regParam': hp.uniform('regParam', 0.01, 0.5),
  'alpha':hp.uniform('alpha', 1.0, 10.0)
  }

# COMMAND ----------

# MAGIC %md Now we can write a function to train and evaluate our model against various hyperparameter values retrieved from our search space:
# MAGIC
# MAGIC **NOTE** We found that setting the *numItemBlocks* and *numUserBlocks* to a value aligned with the number of executors in our cluster helped speed up performance.  We have not performed an exhaustive test of this approach and expect that results will vary with cluster size and specific datasets. 

# COMMAND ----------

# DBTITLE 1,Define Evaluation Function for Tuning Trials
# define model to evaluate hyperparameter values
def evaluate(params):
  
  # clean up params
  if 'maxIter' in params: params['maxIter']=int(params['maxIter'])
  if 'rank' in params: params['rank']=int(params['rank'])
  
  with mlflow.start_run(nested=True):
    
    # instantiate model
    als = ALS(
      rank=100,
      maxIter=20,
      userCol='user_id',  
      itemCol='item_id', 
      ratingCol='rating', 
      implicitPrefs=True,
      numItemBlocks=sc.defaultParallelism,
      numUserBlocks=sc.defaultParallelism,
      **params
      )
    
    # train model
    model = als.fit(ratings_sampled_train)
    
    # generate recommendations
    predicted = (
      model
        .recommendForAllUsers(k)
        .select( 
          'user_id',
          fn.posexplode('recommendations').alias('pos', 'rec') 
          )
        .withColumn('recs', fn.expr("collect_list(rec.item_id) over(partition by user_id order by pos)"))
        .groupBy('user_id')
          .agg( fn.max('recs').alias('recs'))
        .withColumn('prediction', fn.col('recs').cast('array<double>'))
      )
    
    # score the model 
    eval = RankingEvaluator( 
      predictionCol='prediction',
      labelCol='label',
      metricName='precisionAtK',
      k=k
      )
    mapk = eval.evaluate( predicted.join(actuals, on='user_id') )
    
    # log parameters & metrics
    mlflow.log_params(params)
    mlflow.log_metrics( {'map@k':mapk} )
    
  return {'loss': -1 * mapk, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md There's a lot going on in our function, so let's break it down a bit.  First, we receive our hyperparameter values from the search space.  Because some of the values are selected from a range, they come in as floats when they are expected to be integers, so we've got a little clean up to do.
# MAGIC
# MAGIC We then train our model using these hyperparameter values.  The trained model is used to generate predictions which are then evaluated to get a MAK@K score.  We log the evaluation metric and the hyperparameter values that lead to that to mlflow so we can examine these later.  And we then return our evaluation metric back to hyperopt for this run.  Notice that hyperopt is expecting a loss value to minimize.  Because MAP@K is better as it increases, we just flip it to a negative value to get hyperopt to work with it properly.
# MAGIC
# MAGIC Notice too that we are referencing some variables such as *k* and *actuals* that aren't defined in the function.  We'll define these now to make them accessible to our function: 

# COMMAND ----------

# DBTITLE 1,Define Variables Used Between Runs
# k for map at k evaluations
k = 10

# calculate actuals once and cache for faster evaluations
actuals = (
  ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(item_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>'))
  ).cache()

# COMMAND ----------

# MAGIC %md Now we can bring everything together to perform our training runs.  Here we ask hyperopt to use our training function to evaluate values from our search space.  With each of the evaluation cycles, hyperopt considers the results and adjusts its search to hone in on an optimal set of hyperparameter values.  
# MAGIC
# MAGIC Notice that we are using *Trails()* and not *SparkTrails()* with our hyperopt run.  *SparkTrails()* will attempt to parallelize our hyperparameter tuning runs across a Databricks cluster, but we are already making use of the Spark MLLib ALS model which is itself distributed.  You can only employ one parallelization pattern at a time so we leverage distributed model training and hyperopt run once cycle at a time across our cluster:

# COMMAND ----------

# DBTITLE 1,Tune the Model
# disable model logging at this phase
mlflow.autolog(exclusive=False, log_models=False) # https://docs.databricks.com/mlflow/databricks-autologging.html

# perform training runs
with mlflow.start_run(run_name='als_hyperopt_run'):
  
  argmin = fmin(
    fn=evaluate,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=Trials()
    )
  
# report on best parameters discovered
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md As mentioned earlier, we are logging each trail to mlflow.  In Databricks, logging is enabled by default with hyperopt though we can configure it to not record the model itself as we did in the code above.  
# MAGIC
# MAGIC Clicking on the mlflow tracking (flask) icon towards the top-right of the notebook and then selecting the view experiments icon in the upper right of the resulting pane, we can see details about the different model runs and perform comparisons to understand how different parameters affect our evaluation metrics.  More details on this can be found [here](https://docs.databricks.com/mlflow/tracking.html), but this data can be used to help us narrow our search space so that future iterations can spend more time scrutinizing the most productive regions of the space:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/als_mlflow_alpha2.PNG'>

# COMMAND ----------

# MAGIC %md ##Step 4: Train and Persist Final Model
# MAGIC
# MAGIC With our hyperparameters selected, we can now proceed to perform a final training run against our full set of data and then persist our model for later use. Notice that we are using the full set of ratings data and evaluating over the same.  Its important that our model have full access to implicit ratings available to us as we are limited in our ability to generate ratings for users and products on which the model is trained. In this regard, our model is really more of a transformation than a true predictive model capable of making generalizations:

# COMMAND ----------

# DBTITLE 1,Train Final Model
k = 10

actuals = (
  ratings
    .withColumn('selections', fn.expr("collect_list(item_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>'))
  ).cache()

# get parameters from prior tuning run
params = space_eval(search_space, argmin)
if 'maxIter' in params: params['maxIter']=int(params['maxIter'])
if 'rank' in params: params['rank']=int(params['rank'])

with mlflow.start_run(run_name='als_full_model'):

  # instantiate model
  als = ALS(
    rank=100,
    maxIter=50,
    userCol='user_id',  
    itemCol='item_id', 
    ratingCol='rating', 
    implicitPrefs=True,
    numItemBlocks=sc.defaultParallelism,
    numUserBlocks=sc.defaultParallelism,
    **params
    )

  # train model
  model = als.fit(ratings)

  # generate recommendations
  predicted = (
    model
      .recommendForAllUsers(k)
      .select( 
        'user_id',
        fn.posexplode('recommendations').alias('pos', 'rec') 
        )
      .withColumn('recs', fn.expr("collect_list(rec.item_id) over(partition by user_id order by pos)"))
      .groupBy('user_id')
        .agg( fn.max('recs').alias('recs'))
      .withColumn('prediction', fn.col('recs').cast('array<double>'))
    )

  # perform evaluation
  eval = RankingEvaluator( 
    predictionCol='prediction',
    labelCol='label',
    metricName='precisionAtK',
    k=k
    )
  
  mapk = eval.evaluate( predicted.join(actuals, on='user_id') )

  # log model details
  mlflow.log_params(params)
  mlflow.log_metrics( {'map@k':mapk} )
  # mlflow.spark.log_model(model, artifact_path='model', registered_model_name=config['model name'])

# COMMAND ----------

# MAGIC %md Our model is now registered with mlflow using the name *als*.  With each run of the cell above, a new version of the model is registered.  The version number can be used to allow us to track the model as it moves through subsequent consideration for a production deployment.  We'll retrieve that version number here before then moving on to deployment steps in the next notebook:

# COMMAND ----------

# DBTITLE 1,Get Persisted Model Version Number
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{config['model name']}'")[0].version

model_version