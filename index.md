## Introduction

In Azure machine learning, datasets as the assets can be retrieved when we perform machine learning assignments. It is a fundamental element in any machine learning workload. In this article, we will discuss how to create and manage datastores and datasets in workspace.

## 1. Some definitions

In Azure machine learning, datastores are abstractions for cloud data sources, there are several types of datastores:

1. Azure Storage(blob and file containers)
2. Azure Data Lake stores
3. Azure SQL database
4. Azure Databricks file system(DBFS)

Noting that every workspace has two built-in datastores, they are an Azure Storage blob container, and an Azure Storage file container that are used as system storage by Azure machine learning. So we also can add a third datastore that gets added to your workspace.

In most machine learning projects, you will likely need to work with data sources of your own - either because you need to store larger volumes of data than the built-in datastores support, or because you need to integrate your machine learning solution with data from existing applications.

## 2. Using datastores

To add a datastore to your workspace, you can register it using the graphical interface in Azure Machine Learning studio, or you can use the Azure Machine Learning SDK. For example, the following code registers an Azure Storage blob container as a datastore named blob_data.

```python
from azureml.core import Workspace, Datastore

ws = Workspace.from_config()

# Register a new datastore
blob_ds = Datastore.register_azure_blob_container(workspace=ws, 
                                                  datastore_name='blob_data', 
                                                  container_name='data_container',
                                                  account_name='az_store_acct',
                                                  account_key='123456abcde789…')
```
You can view and manage datastores in the studio or use the SDKs to list the names of each datastore in the workspace:

```python
for ds_name in ws.datastores:
    print(ds_name)
```
Except two default datastore, there are a added datastore now:

![image](https://user-images.githubusercontent.com/71245576/116011586-dc80da00-a5f3-11eb-9e59-ea885dd06009.png)

You also can get a reference to any datastore by using the Datastore.get() method:

```python
![image](https://user-images.githubusercontent.com/71245576/116011599-ef93aa00-a5f3-11eb-83a0-581cb021b29f.png)

```
In addition, get the default datastore:
```python
blob_store = Datastore.get(ws, datastore_name='blob_data')
```

There are some considerations that are deserved concerning:

1. When using Azure blob storage, premium level storage may provide improved I/O performance for large datasets. However, this option willl increase cost and may limit replication options for data redundancy.
2. When working with data files, although CSV format is common, Parquet format generally results in better performance
3. You can access any datastore by name, but you may want to consider changing the default datastore (which is initially the built-in workspaceblobstore datastore).

To change the default datastore, use the set_default_datastore() method:

```python
ws.set_default_datastore('blob_data')
```

## 3. Using datasets

In Azure, it is very useful that datasets are versioned packaged data objects that can be consumed in experiments and pipelines. Datasets are the recommended way to work with data, and are the primary mechanism for advanced Azure Machine Learning capabilities like data labeling and data drift monitoring.

There are two types of datasets:

![image](https://user-images.githubusercontent.com/71245576/116011759-aabc4300-a5f4-11eb-903e-ad8efe180e0d.png)

Now let's try to create and register datasets. The first is to create and register tabular datasets:

You should use the from_delimited_files method:
```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
```
Now you can go back to check if the dataset is registered:

![image](https://user-images.githubusercontent.com/71245576/116012263-9299f300-a5f7-11eb-9971-ce818f9e1d1d.png)

To create a file dataset using the SDK, use the from_files method of the Dataset.File class, like this:
```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
```
Check that this img_files file is registered.

Now let's retrieve the dataset, there are two methods: the datasets dictionary attribute and get_by_name or get_by_id memthod pf the Dataset class.

```python
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')
```

## 4. Dataset versioning

Datasets can be versioned, enabling you to track historical versions of datasets that were used in experiments, and reproduce those experiments with data in the same state.

You can create a new version of a dataset by registering it with the same name as a previously registered dataset and specifying the create_new_version property:

```python
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
```

Review the versions of this dataset: There are two versions

![image](https://user-images.githubusercontent.com/71245576/116012543-0c7eac00-a5f9-11eb-9c22-3589514ad3f9.png)

Now we can retrieve a specific version of a dataset by specifying the version parameter in the get_by_name method of the Dataset class:

```python
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)
```

## 5. Working with datasets

Datasets in Azure are the primary way to pass data to experiments that train models. You can read data directly from a tabular dataset by converting it into a Pandas or Spark dataframe:

```python
df = tab_ds.to_pandas_dataframe()
# code to work with dataframe goes here, for example:
print(df.head())
```

But if you want to pass a dataset as a script argument. You need to config it and script it. There are two types of datasets: tabular dataset and file datasets, let's discess about these.

### 5.1 Working with tabular datasets

You can use a script argument or a named input for a tabular dataset to pass data. When you pass a tabular dataset as a script argument, the argument received by the script is the unique ID for the dataset in your workspace. You can then get the workspace from the run context and use it to retrieve the dataset by it's ID.

First you need to save the script to the specified directory with the datasets that you will use, see the script:
```python
from azureml.core import Run, Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='dataset_id', default='9947fd35-91cd-4dae-a587-c773dcbef9f0')
args = parser.parse_args(args=[])

run = Run.get_context()
ws = run.experiment.workspace
dataset = Dataset.get_by_id(ws, id=args.dataset_id)
data = dataset.to_pandas_dataframe()
```

Use the ScriptRunConfig to pass dataset ID to script:
```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds],
                                environment=env)
```

You can submit it now like this:
```python
experiment = Experiment(workspace = ws, name = 'my-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
```

If you submit it, wait a while, when the experiment has completed, review the run like this:

![image](https://user-images.githubusercontent.com/71245576/116154465-b1ab8a00-a6b6-11eb-8778-034598efc46b.png)

I should notice you that you should use .py file, if you use .ipynb file it will occur a bug that said local execulation failed.

Now let's use a named input for a tabular dataset:

You can pass a tabular dataset as a named input. You in this approach use the as_named_input method of the dataset to specify a name for the dataset. Then in the script you can retrieve the dataset by name from the run context's input_datasets collection without needing to retrieve it from the workspace. If using this appraoach you still need to include a script argument for the dataset even though you do not actually use it to retrieve the dataset.

See the script:
```python
from azureml.core import Run
from azureml.core import Run, Dataset
import azureml.core
from azureml.core import Workspace, Dataset
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_id')
args = parser.parse_args(args=[])

run = Run.get_context()
dataset = run.input_datasets['my_dataset']
data = dataset.to_pandas_dataframe()
```


See the ScriptRunConfig:
```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', tab_ds.as_named_input('my_dataset')],
                                environment=env)
```
Submit and it will run a few seconds. Wait it for finishment.

![image](https://user-images.githubusercontent.com/71245576/116158364-89268e80-a6bc-11eb-9700-03ac913c03ae.png)

### 5.2 Passing file dataset to an experiment script

There are also two approaches, either use a script argument or a named input. For using a script argument for a file dataset. Unlike with a tabular dataset you must specify a mode for the file dataset argument which can be as_download or as_mount. This provides an access point that the script can use to read the files in the dataset. 

In most cases, you should use as_download, which copies the files to a temporary location on the compute where the script is being run. However, if you are working with a large amount of data for which there may not be enough storage space on the experiment compute, use as_mount to stream the files directly from their source.

I found there are some bugs in the tutorial code, so I debugged and refined it:
```python
from azureml.core import Run
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args(args=[])
run = Run.get_context()

imgs = glob.glob("args.ds_ref +'/*.jpg'")
```

The congire file ScriptRunConfig:
```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_download()],
                                environment=env)
```

Now let's use a named input for a file dataset:

In this approach, you use the as_named_input method of the dataset to specify a name before specifying the access mode. Then in the script, you can retrieve the dataset by name from the run context's input_datasets collection and read the files from there. As with tabular datasets, if you use a named input, you still need to include a script argument for the dataset, even though you don’t actually use it to retrieve the dataset.

The script has been refined(the script that the tutorial provided has bugs):

```python
from azureml.core import Run
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_ref')
args = parser.parse_args()
run = Run.get_context()

dataset = run.input_datasets['my_ds']
imgs= glob.glob("dataset + '/*.jpg'")
```

The configure file ScriptRunConfig:
```python
env = Environment('my_env')
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults',
                                                  'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                arguments=['--ds', file_ds.as_named_input('my_ds').as_download()],
                                environment=env)
```
See it has completed:

![image](https://user-images.githubusercontent.com/71245576/116160808-b2e1b480-a6c0-11eb-8d08-8a6a5152160e.png)


## Reference:

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
