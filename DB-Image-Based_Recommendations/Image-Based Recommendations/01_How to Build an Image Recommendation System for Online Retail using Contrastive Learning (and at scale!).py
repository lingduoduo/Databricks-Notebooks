# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/image-based-recommendations. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md
# MAGIC # How to Build an Image Recommendation System for Online Retail using Contrastive Learning (and at scale!)

# COMMAND ----------

# MAGIC %md
# MAGIC In this article, you will learn the end to end process of building a recommender that uses a model trained using similarity learning, a novel machine learning approach more suitable for finding similar items. You will use the Tensorflow_similarity library to train the model and Spark,Horovod, and Hypeopt, to scale the model training across a GPU cluster. Mlflow will be used to log and track all aspects of the process and Delta will be used to preserve data lineage and reproducibility.
# MAGIC
# MAGIC At a high level, similarity models are trained using contrastive learning. In contrastive learning, the goal is to make the machine learning model (an adaptive algorithm) learn an embedding space where the distance between similar items is minimized and distance between dissimilar items is maximized. In this quickstart we will use the fashion MNIST dataset, which comprises of around 70,000 images of various clothing items. Based on the above description, a similarity model trained on this labelled dataset will learn an embedding space where embeddings of similar items e.g. boots are closer together and different items e.g. boots and bandanas are far apart. 
# MAGIC
# MAGIC This could be illustrated as below.

# COMMAND ----------

displayHTML("<img src='https://github.com/avisoori-databricks/Databricks_image_recommender/blob/main/images/simrec_embed.png?raw=true'")

# COMMAND ----------

# MAGIC %md 
# MAGIC In similarity learning, the goal is to teach the model to discover a space where the similar items are grouped closer to each other and dissimilar items are separated even more. In supervised similarity learning, the algorithms has access to image labels to learn from, in addition to the image data itself.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table of Contents
# MAGIC
# MAGIC - 1	 Set up: We will look at cluster creation, installing the necessary libraries, importing the required modules and getting the data into Delta tables for subsequent tasks.
# MAGIC - 2	 Model training: We will take a look at training a Similarity model. We draw from the examples in the official Tensorflow Similarity repository.
# MAGIC - 3	 Scaling Hyperparameter search with Hyperopt and Spark: One way of scaling is to distribute the search for the best hyperparameter combination leading to optimal model performance.
# MAGIC - 4	 Scaling model training with Horovod and Spark: If the data sets are large, we can scale training of a single model across a cluster.
# MAGIC - 5  Model Deployment - We will look at deploying this model as a REST endpoint using MLflow serving and post processing logic in the context of an application.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up
# MAGIC
# MAGIC Enter some details about the cluster configurations (i.e. GPU cluster with 2 or more worker nodes to leverage distributed compute). T4 GPUs are a good choice for this task.

# COMMAND ----------

displayHTML("<img src='https://github.com/avisoori-databricks/Databricks_image_recommender/blob/main/images/simrec_gpu.png?raw=true'>")

# COMMAND ----------

# MAGIC %md
# MAGIC Install the Tensorflow Similarity library:

# COMMAND ----------

# MAGIC %pip install tensorflow_similarity protobuf==3.20.3

# COMMAND ----------

# MAGIC %md
# MAGIC Perform the required imports

# COMMAND ----------

import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import mlflow
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam, Adadelta
from pyspark.ml.feature import OneHotEncoder

    
import tensorflow_similarity as tfsim
from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.layers import MetricEmbedding
from tensorflow_similarity.losses import MultiSimilarityLoss  
from tensorflow_similarity.models import SimilarityModel 
from tensorflow_similarity.samplers import MultiShotMemorySampler 
from tensorflow_similarity.samplers import select_examples 
from tensorflow_similarity.visualization import viz_neigbors_imgs 
from tensorflow_similarity.visualization import confusion_matrix 

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/image_recommender"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md
# MAGIC Note the version of the Tensorflow and Tensorflow similarity libraries that are installed.

# COMMAND ----------

print('TensorFlow:', tf.__version__)
print('TensorFlow Similarity', tfsim.__version__)


# COMMAND ----------

# MAGIC %md 
# MAGIC Run the two cells below to fetch the data from the official fashion MNIST repo by Zalando (cited in the blog) and create the Delta tables

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /databricks/driver
# MAGIC wget -O  test_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz
# MAGIC wget -O  test_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz
# MAGIC wget -O  train_images.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz
# MAGIC wget -O  train_labels.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz
# MAGIC
# MAGIC
# MAGIC gunzip -dk *.gz
# MAGIC
# MAGIC rm -r train_labels.gz test_labels.gz train_images.gz test_images.gz
# MAGIC
# MAGIC ls

# COMMAND ----------

# MAGIC %md
# MAGIC The function below, adapted from https://pjreddie.com/projects/mnist-in-csv/, is for converting the datasets downloaded above to delta tables corresponding to train and test images.

# COMMAND ----------


datasets = [['test_images', 'test_labels','/FileStore/tables/user/delta/fmnist_test_data', 10000],  ['train_images', 'train_labels', '/FileStore/tables/user/delta/fmnist_train_data', 60000]]


def convert(imgf, labelf, outf, n):
  """This accepts an image file name, label file name, an output path and number of records. It reads and converts the image data into a Delta table in the specified path"""
  f = open(f"/databricks/driver/{imgf}", "rb")
  l = open(f"/databricks/driver/{labelf}", "rb")

  f.read(16)
  l.read(8)
  images = []

  for i in range(n):
      image = [ord(l.read(1))]
      for j in range(28*28):
          image.append(ord(f.read(1)))
      images.append(image)

  f.close()
  l.close()
  df  = pd.DataFrame(images)
  sparkdf = spark.createDataFrame(df)
  sparkdf.write.format('delta').mode('overwrite').save(outf)
    

# COMMAND ----------

for dataset in datasets:
  convert(dataset[0], dataset[1], dataset[2], dataset[3])

# COMMAND ----------

# MAGIC %md
# MAGIC The classes in the fashion mnist dataset are as follows. 
# MAGIC
# MAGIC - Label	Description
# MAGIC - 0	 T-shirt/top
# MAGIC - 1	 Trouser
# MAGIC - 2	 Pullover
# MAGIC - 3	 Dress
# MAGIC - 4	 Coat
# MAGIC - 5	 Sandal
# MAGIC - 6	 Shirt
# MAGIC - 7	 Sneaker
# MAGIC - 8	 Bag
# MAGIC - 9	 Ankle boot

# COMMAND ----------

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# COMMAND ----------

# MAGIC %md Read the Delta tables into train and test datasets

# COMMAND ----------

train = spark.read.format("delta").load("/FileStore/tables/user/delta/fmnist_train_data").toPandas().values
test = spark.read.format("delta").load("/FileStore/tables/user/delta/fmnist_test_data").toPandas().values

# COMMAND ----------

# MAGIC %md Define a function to shape the image data to a form that the model training process could accommodate. This function is entirely a function of what your image data looks like and what model architecture you choose

# COMMAND ----------

def get_dataset(train, test, rank=0, size=1):
  from tensorflow import keras
  import numpy as np
  
  np.random.shuffle(train)
  np.random.shuffle(test)

  x_train = train[:, 1:].reshape(-1, 28, 28)
  y_train = train[:, 0].astype(np.int32)
  x_test = test[:, 1:].reshape(-1, 28, 28)
  y_test = test[:, 0].astype(np.int32)

  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255.0
  x_test /= 255.0
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Define the model architecture. The beauty of similarity learning is that you can observe significantly robust performance with relatively simple convolutional neural network architecture

# COMMAND ----------

def get_model():
    from tensorflow_similarity.layers import MetricEmbedding
    from tensorflow.keras import layers
    from tensorflow_similarity.models import SimilarityModel
    
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1/255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    # tune the embedding size to tune the look up times. Smaller embeddings will result in quicker look up times but less accurate results. The converse is true for larger embeddings.
    outputs = MetricEmbedding(256)(x)
    return SimilarityModel(inputs, outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the machine learning model on a single GPU and track model parameters/ metrics

# COMMAND ----------

#Number of overall classes in the dataset is 10
num_classes = 10

# COMMAND ----------

# MAGIC %md Define a function for training the model with the architecture and datasets prescribed above

# COMMAND ----------

def train_model(train, test, learning_rate=0.001):
  """This function encapsulates the general training logic for a similarity model with Tensorflow Similarity"""
  from tensorflow import keras
  from tensorflow_similarity.losses import MultiSimilarityLoss   
  from tensorflow_similarity.samplers import MultiShotMemorySampler
  from tensorflow.keras.optimizers import Adam
  import mlflow
  mlflow.tensorflow.autolog()
  #The number of classes in the Fashion MNIST dataset is 10
  (x_train, y_train), (x_test, y_test) = get_dataset(train, test)
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  #The number of classes used to train the model. The idea is that a similarity model can generalize well into hitherto unseen classes. 
  #So 6 out of the 10 classes in the Fashion MNIST dataset will be used to train the model.
  num_classes_ = 6   
  class_per_batch = num_classes_
  example_per_class = 6  
  epochs = 10
  steps_per_epoch = 1000  

  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=class_per_batch, 
                                   examples_per_class_per_batch=example_per_class,
                                   class_list=classes[:num_classes_],  
                                   steps_per_epoch=steps_per_epoch)
  model = get_model()
  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  model.compile(optimizer=Adam(learning_rate), loss=loss)
  model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))
  return model 

# COMMAND ----------

model = train_model(train, test, learning_rate=0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train across a cluster of GPUs with Horovod and Spark

# COMMAND ----------

import os
import time

# Remove any existing checkpoint files
dbutils.fs.rm(("/avi_ml/MNISTDemo/train"), recurse=True)

# Create directory
checkpoint_dir = '/dbfs/avi_ml/MNISTDemo/train/{}/'.format(time.time())
os.makedirs(checkpoint_dir)
print(checkpoint_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC Define a function to train a single model in a distributed manner with Horovod

# COMMAND ----------

def train_hvd(train, test, checkpoint_path, learning_rate=0.001):
  """This function encapsulates the logic necessary to distribute model training across a cluster with Horovod and Spark. More details can be found at https://databricks.github.io/spark-deep-learning/index.html"""
  #Encapsulate all the imports and the logic for training the model within this function 
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam
  #Horovod flavor matters
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  import mlflow
  
  
  
  from tensorflow_similarity.utils import tf_cap_memory
  from tensorflow_similarity.layers import MetricEmbedding  
  from tensorflow_similarity.losses import MultiSimilarityLoss   
  from tensorflow_similarity.models import SimilarityModel  
  from tensorflow_similarity.samplers import MultiShotMemorySampler  
  from tensorflow_similarity.samplers import select_examples  
  from tensorflow_similarity.visualization import viz_neigbors_imgs   
  from tensorflow_similarity.visualization import confusion_matrix  
  
  
  
  # Initialize Horovod
  hvd.init()

  
  batch_size = 128

  # Pin GPU to be used to process local rank (one GPU per process)
  # These steps are skipped on a CPU cluster
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
  (x_train, y_train), (x_test, y_test) = get_dataset(train, test, hvd.rank(), hvd.size())
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  num_classes_ = 6  
  class_per_batch = num_classes_
  example_per_class = 6 
  epochs = 30
  steps_per_epoch = 1000 
  
  
  model = get_model()
  
  # Adjust learning rate based on number of GPUs
  optimizer = Adadelta(learning_rate=learning_rate * hvd.size())

  # Use the Horovod Distributed Optimizer
  optimizer = hvd.DistributedOptimizer(optimizer)
  
  

  # Create a callback to broadcast the initial variable states from rank 0 to all other processes.
  callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
  # This is required to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
 
  
  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=class_per_batch, 
                                   examples_per_class_per_batch=example_per_class,
                                   class_list=classes[:num_classes_], 
                                   steps_per_epoch=steps_per_epoch)
  
  

  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  

  
  
  model.compile(optimizer=Adam(learning_rate), loss=loss)
  


  # Save checkpoints only on worker 0 to prevent conflicts between workers
  if hvd.rank() == 0:
      mlflow.tensorflow.autolog()
      

      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))
      
  model.fit(sampler, callbacks = callbacks ,epochs=epochs, validation_data=(x_test, y_test))

# COMMAND ----------

# DBTITLE 1,Set up mlflow experiment in the user's personal workspace folder
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/image_recommendation"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# Start the training process and mlflow will record logged metrics and parameters
with mlflow.start_run() as run:  
  from sparkdl import HorovodRunner
  from tensorflow_similarity.losses import MultiSimilarityLoss  # specialized similarity loss
  import mlflow

  checkpoint_path = checkpoint_dir + '/checkpoint-{epoch}.ckpt'
  learning_rate = 0.001
   
  
  # Run HorovodRunner
  hr = HorovodRunner(np=2, driver_log_verbosity='all')

  hr.run(train_hvd,  train  = train, test = test, checkpoint_path=checkpoint_path, learning_rate=learning_rate)
  
  distance = 'cosine' 

  loss = MultiSimilarityLoss(distance=distance)

  hvd_model = get_model()

  optimizer=Adam(learning_rate)

  hvd_model.compile(optimizer=Adam(learning_rate), loss=loss)

  hvd_model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))

  (x_train, y_train), (x_test, y_test) = get_dataset(train, test)

  score = hvd_model.evaluate(x_test, y_test, verbose=0)

  mlflow.log_metric("loss", score)
  mlflow.log_param("lr", learning_rate)

  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Hyperparameter optimization with Hyperopt and Spark

# COMMAND ----------

# MAGIC %md Define a training function to use with hyperopt. This is very similar to the training functions defined earlier. Here, the difference is multiple models are trained across a cluster where at any given point each executor is training a single model with a unique combination of hyperparameters

# COMMAND ----------

def train_hyperopt(space):
  """ This function accepts a dictionary which represents the space of hyperparameters we want the Bayesian search to take place in. More information about Hyperopt can be found at: http://hyperopt.github.io/hyperopt/scaleout/spark/ """
  from tensorflow import keras
  from tensorflow_similarity.losses import MultiSimilarityLoss  # specialized similarity loss
  from tensorflow_similarity.samplers import MultiShotMemorySampler
  from tensorflow.keras.optimizers import Adam
  import mlflow
  
  mlflow.tensorflow.autolog()


  (x_train, y_train), (x_test, y_test) = get_dataset(train, test )
  classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
  num_classes_ = 7
  classes_per_batch = num_classes_
  examples_per_class = space['examples_per_class'] 
  epochs = 10
  steps_per_epoch = space['steps_per_epoch'] 

  sampler = MultiShotMemorySampler(x_train, y_train, 
                                   classes_per_batch=classes_per_batch, 
                                   examples_per_class_per_batch=examples_per_class,
                                   class_list=classes[:num_classes_], 
                                   steps_per_epoch=steps_per_epoch)
  model = get_model()
  distance = 'cosine' 
  loss = MultiSimilarityLoss(distance=distance)
  model.compile(optimizer=Adam(space["learning_rate"]), loss=loss)
  model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))
  return model.evaluate(x_test, y_test)

# COMMAND ----------

# MAGIC %md Import the necessary packages and define the hyperparameter search space as a dictionary

# COMMAND ----------

import numpy as np
from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

space = {
  'steps_per_epoch': hp.choice('steps_per_epoch', np.arange(100, 2000, 250, dtype=int)),
  'examples_per_class' : hp.choice('examples_per_class',np.arange(5, 10, 1, dtype=int)),
  'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1))
}

# COMMAND ----------

# MAGIC %md 
# MAGIC Create an instance of Sparktrials with parallelism of 2 as we only have two workers in this cluster. Change this value according to the number of workers in your cluster.

# COMMAND ----------

import mlflow
trials = SparkTrials(2)

# COMMAND ----------

# MAGIC %md Define the hyperparameter search algorithm and start the search. Because you're using SparkTrials as defined above you're doing this in a distributed manner across your spark cluster.

# COMMAND ----------

algo=tpe.suggest
 
with mlflow.start_run():
  best_params = fmin(
    fn=train_hyperopt,
    space=space,
    algo=algo,
    max_evals=32,
    trials = trials,
  )

# COMMAND ----------

# MAGIC %md Figure out the best parameters that were discovered by the above process. Please note that steps_per_epoch == 4 indicates the 4th index of the numpy range passed, which is 1100

# COMMAND ----------

print(best_params)

# COMMAND ----------

# MAGIC %md 
# MAGIC Use these parameters to train a model to build an index, which will then be used to querying.

# COMMAND ----------

#Train final model 
(x_train, y_train), (x_test, y_test) = get_dataset(train, test)
classes = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
num_classes = 7  
classes_per_batch = num_classes
example_per_class = 20
epochs = 20
steps_per_epoch = 1100
learning_rate = 0.0013508067254937172

sampler = MultiShotMemorySampler(x_train, y_train, 
                                 classes_per_batch=classes_per_batch, 
                                 examples_per_class_per_batch=example_per_class,
                                 class_list=classes[:num_classes],
                                 steps_per_epoch=steps_per_epoch)
tfsim_model = get_model()
distance = 'cosine' 
loss = MultiSimilarityLoss(distance=distance)
tfsim_model.compile(optimizer=Adam(learning_rate), loss=loss)
tfsim_model.fit(sampler, epochs=epochs, validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md 
# MAGIC Inspect the architecture of the model.

# COMMAND ----------

tfsim_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build an Index

# COMMAND ----------

x_index, y_index = select_examples(x_train, y_train, classes, 20)
tfsim_model.reset_index()
tfsim_model.index(x_index, y_index, data=x_index)

# COMMAND ----------

# MAGIC %md Let's inspect one of the images 

# COMMAND ----------

from matplotlib import pyplot as plt

sample_image = x_index[0]
sample_image = sample_image.reshape(1, sample_image.shape[0], sample_image.shape[1]) 
plt.imshow(sample_image[0], interpolation='nearest')
plt.show()

# COMMAND ----------

# MAGIC %md Let's perform a quick sanity check along the way. The label corresponding to this image

# COMMAND ----------


label  = y_index[0]
label
#4 is a Coat as indicated here: https://github.com/zalandoresearch/fashion-mnist

# COMMAND ----------

# MAGIC %md 
# MAGIC Test what the recommendations look like for a given image. The model object will return the n approximately nearest neighbors based on the index in this case.

# COMMAND ----------

#First I inspect the type of the returned objects stored in variables x_display and y_display
x_display, y_display = select_examples(x_test, y_test, classes, 1)

type(x_display), type(y_display)

# COMMAND ----------

# select
x_display, y_display = select_examples(x_test, y_test, classes, 1)

# lookup nearest neighbors in the index
nns = np.array(tfsim_model.lookup(x_display, k=5))

# display
for idx in np.argsort(y_display):
    viz_neigbors_imgs(x_display[idx], y_display[idx], nns[idx], 
                      fig_size=(16, 2), cmap='Greys')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an MLflow Pyfunc wrapper class for deployment and querying 

# COMMAND ----------

# MAGIC %md Save the model to a specified directory. This saves the model itself and the index which we use for querying.

# COMMAND ----------

tfsim_path = "/databricks/driver/models/tfsim.pth"

tfsim_model.save(tfsim_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Create an `artifacts` dictionary that assigns a unique name to the saved tensorflow_similarity model file.
# MAGIC This dictionary will be passed to `mlflow.pyfunc.save_model`, which will copy the model file
# MAGIC into the new MLflow Model's directory.

# COMMAND ----------

artifacts = {
    "tfsim_model": tfsim_path
}

# COMMAND ----------

# Define the custom model class
import mlflow.pyfunc
class TfsimWrapper(mlflow.pyfunc.PythonModel):
    """ model input is a single row, single column pandas dataframe with base64 encoded byte string i.e. of the type bytes. Column name is 'input' in this case"""
    """ model output is a pandas dataframe where each row(i.e.element since only one column) is a string  converted to hexadecimal that has to be converted back tobytes and then a numpy array using np.frombuffer(...) and reshaped to (28, 28) and then visualized (if needed)"""
    
    def load_context(self, context):
      import tensorflow_similarity as tfsim
      from tensorflow_similarity.models import SimilarityModel
      from tensorflow.keras import models
      import pandas as pd
      import numpy as np
      
      
      self.tfsim_model = models.load_model(context.artifacts["tfsim_model"])
      self.tfsim_model.load_index(context.artifacts["tfsim_model"])

    def predict(self, context, model_input):
      from PIL import Image
      import base64
      import io

      image = np.array(Image.open(io.BytesIO(base64.b64decode(model_input["input"][0].encode()))))    
      #The model_input has to be of the form (1, 28, 28)
      image_reshaped = image.reshape(-1, 28, 28)/255.0
      images = np.array(self.tfsim_model.lookup(image_reshaped, k=5))
      image_dict = {}
      for i in range(5):
        image_dict[i] = images[0][i].data.tostring().hex()
        
      return pd.DataFrame.from_dict(image_dict, orient='index')
      
    


# COMMAND ----------

from sys import version_info

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

# COMMAND ----------

# MAGIC %md 
# MAGIC Create a Conda environment for the new MLflow Model that contains all necessary dependencies and the correct python version.

# COMMAND ----------

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'tensorflow_similarity=={}'.format(tfsim.__version__),
          'tensorflow_cpu ==2.7.0',
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'tfsim_env'
}

# COMMAND ----------

# Save the MLflow Model
mlflow_pyfunc_model_path = "/databricks/driver/models/tfsim_mlflow.pth"
mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path, python_model=TfsimWrapper(), artifacts=artifacts,
        conda_env=conda_env)

# COMMAND ----------

# MAGIC %md 
# MAGIC You can read more about how to create custom models here: https://mlflow.org/docs/latest/models.html#custom-python-models 

# COMMAND ----------

# MAGIC %md
# MAGIC Test if the predict method of the pyfunc wrapper actually works

# COMMAND ----------

img = "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lEQVR4nF3Sz2vaYBgH8DdKNypDZYO2hxWnVDGiESMxIUoStGKCMWgwEpVEtKIyxR848Qet1FEHPRShp0FPu4wNdtoOg3Wn3Xvb/7Npq0Z93hBe3g/PA3m/AUCrkG/xrlkAtKjlIQR086359ejhz93v+wcMfr7RMVf9MV8dpKnuz7vk6WQ2Fl9t2GGqd95tdduicqoOzwqN/hWxxv1gnPZ5vAjK8HyE8FEwglG2lVqJYEgQRUlMyUI5+fFvOJvA1q0kwpbYVEVMJVg+o0TlYorHGMMSE0ikwIfnYzEy6EcJNJgs47HV3Cjlm7QtVpfD5nScwLDHySs0Y1yiy2L78cWBIH6MwQM46ie+dgJh8slO/n0yz+4LMstyUZrj43L1O28djp9QHxOOPv9631EFlqE4qV6pfSuH8JdLPHa8UUdxNR4O00GKkXIZF2w52t+4wjJBu+0exGGHvUhEr53rFhk0qzLqQ2GnFyNSkgnSr2KBFoudvsso+WwmW8mfN4xgHdncAESqMSrCEAQpcErRAOk2MwMMQwSCATJEhDxkZE8L+xHVNIOSbjcM4yhV3wOaLUq47lebhayUrTfqt6YthABeyFXOykVVzsqcbAa6LeTSASZAzm8W9WOJgy0EoPchX8uXJFFQ8uXLwx3sTCeXF9Pm29ZgMOiZgF6bOX9ihVa9NO7k6oqqJJ5BO50Nnk4nJYKLRviG4fFP1qp/O51dXYza18Puzc2LnU/R2zHBTcJev81mNS7tP1M4itBUw7AYAAAAAElFTkSuQmCC"

data = {"input": [img] }
sample_image = pd.DataFrame.from_dict(data)

# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# Evaluate the model
test_predictions = loaded_model.predict(sample_image)
print(test_predictions)


# COMMAND ----------

# MAGIC %md 
# MAGIC Deploy the model as a live REST endpoint for querying

# COMMAND ----------

# MAGIC %md
# MAGIC First infer the model signature. More details can be found here: https://www.mlflow.org/docs/latest/models.html#model-signature

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(sample_image, loaded_model.predict(sample_image))

# COMMAND ----------

# MAGIC %md
# MAGIC Log the trained model. Since we have created a custom model wrapper class, this includes both the model and the index

# COMMAND ----------

mlflow.pyfunc.log_model(artifact_path="tfsim", python_model=TfsimWrapper(), artifacts=artifacts,
        conda_env=conda_env, signature = signature)

# COMMAND ----------

# MAGIC %md
# MAGIC Follow the instructions given in the following link: https://docs.databricks.com/applications/mlflow/models.html to deploy this model as a REST endpoint. Refer to the Git repo in the blog post to see a complete example where the model will generate recommendations for any chosen fashion MNIST image. 

# COMMAND ----------

# MAGIC %md 
# MAGIC If you want to test the REST endpoint within the deployment UI in Databricks, use the following example (testing this in the cURL input box within the UI)

# COMMAND ----------

[{"input":"iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAACN0lEQVR4nF3Sz2vaYBgH8DdKNypDZYO2hxWnVDGiESMxIUoStGKCMWgwEpVEtKIyxR848Qet1FEHPRShp0FPu4wNdtoOg3Wn3Xvb\\/7Npq0Z93hBe3g\\/PA3m\\/AUCrkG\\/xrlkAtKjlIQR086359ejhz93v+wcMfr7RMVf9MV8dpKnuz7vk6WQ2Fl9t2GGqd95tdduicqoOzwqN\\/hWxxv1gnPZ5vAjK8HyE8FEwglG2lVqJYEgQRUlMyUI5+fFvOJvA1q0kwpbYVEVMJVg+o0TlYorHGMMSE0ikwIfnYzEy6EcJNJgs47HV3Cjlm7QtVpfD5nScwLDHySs0Y1yiy2L78cWBIH6MwQM46ie+dgJh8slO\\/n0yz+4LMstyUZrj43L1O28djp9QHxOOPv9631EFlqE4qV6pfSuH8JdLPHa8UUdxNR4O00GKkXIZF2w52t+4wjJBu+0exGGHvUhEr53rFhk0qzLqQ2GnFyNSkgnSr2KBFoudvsso+WwmW8mfN4xgHdncAESqMSrCEAQpcErRAOk2MwMMQwSCATJEhDxkZE8L+xHVNIOSbjcM4yhV3wOaLUq47lebhayUrTfqt6YthABeyFXOykVVzsqcbAa6LeTSASZAzm8W9WOJgy0EoPchX8uXJFFQ8uXLwx3sTCeXF9Pm29ZgMOiZgF6bOX9ihVa9NO7k6oqqJJ5BO50Nnk4nJYKLRviG4fFP1qp\\/O51dXYza18Puzc2LnU\\/R2zHBTcJev81mNS7tP1M4itBUw7AYAAAAAElFTkSuQmCC"}]

# COMMAND ----------

# MAGIC %md
# MAGIC This UI can be accessed by following instructions given here: https://docs.databricks.com/applications/mlflow/model-serving.html

# COMMAND ----------

# MAGIC %md
# MAGIC Code and instructions to deploy an app where the REST API endpoint could be used to get recommendations is in the repository at: https://github.com/avisoori-databricks/Databricks_image_recommender/tree/main/recommender_app

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library / data source                  | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | tensorflow                                | package                 | Apache 2.0  | https://github.com/tensorflow/tensorflow/blob/master/LICENSE  |
# MAGIC | fashion-mnist| dataset | MIT | https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE |

# COMMAND ----------

