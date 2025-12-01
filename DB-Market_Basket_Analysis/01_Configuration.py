# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/market-basket-analysis. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md This notebook provides access to consistent configuration settings across the notebooks that make up this solution accelerator.  It also provides instructions for the setup of the files required by these notebooks.

# COMMAND ----------

# DBTITLE 1,Initialize Config Settings
if 'config' not in locals():
  config = {}

# COMMAND ----------

# MAGIC %md The path of the mount point and the folders containing these files are specified as follows:

# COMMAND ----------

# DBTITLE 1,File Path Configurations
config['root_path'] = '/tmp/instacart_market_basket'
config['orders_files'] = config['root_path'] + '/bronze/orders'
config['products_files'] = config['root_path'] + '/bronze/products'
config['order_products_files'] = config['root_path'] + '/bronze/order_products'
config['departments_files'] = config['root_path'] + '/bronze/departments'
config['aisles_files'] = config['root_path'] + '/bronze/aisles'

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
import mlflow
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/market_basket"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.