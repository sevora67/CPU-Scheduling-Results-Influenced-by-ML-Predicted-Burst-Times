# Machine Learning Models for Predicting CPU Burst Times for CPU Schedulers
## Written by: Saul Evora

**Description:** Jupyter Notebook Python program that uses Random Forest (RF), Neural Network (NN), and Support Vector Regression (SVR) to predict CPU burst times. It also compares First Come, First Served (FCFS) and non-preemptive Shortest Job First (SJF) scheduling using actual and predicted runtimes. The program is designed to run in Google Colab.

## Usability

**NOTE:** Before starting the program, make sure you have the dataset named **`processes_datasets.csv`**. The program expects this specific dataset and its job-related columns. Open the Python notebook in **Google Colab**.

### STEPS

**NOTE:** Before executing the program, check the experiment configurations. These are located near the beginning of the notebook and in the `main()` method.

### 1. Enter the experiment settings

The program accepts values for test size, Neural Network epochs, Neural Network batch size, Random Forest trees, and number of jobs.

**EXAMPLE #1:**

```python
# Model settings
NN_BATCH_SIZE = 64
NN_EPOCHS = 25
RF_TREES = 100
TEST_SIZE = 0.20

# Main method
def main():
    number_of_jobs = 4000
```

**What are the settings supposed to mean?**

- **Test size:** `0.20` means 20% of the selected jobs are used for testing and the remaining 80% for training.
- **Epochs:** `25` is the maximum number of training epochs for the Neural Network. Early stopping may end training sooner.
- **Batch size:** `64` means the Neural Network trains on batches of 64 observations at a time.
- **Random Forest trees:** `100` means the Random Forest uses 100 decision trees.
- **Number of jobs:** `4000` means the program selects the earliest 4,000 valid jobs before dividing them into training and testing sets. With the settings above, 3,200 jobs are used for training and 800 for testing.

**NOTE:** Change the settings as needed for your experiment. Record the settings used for each run when comparing results.

### 2. Choose ONLY ONE method to import the dataset

The program provides two alternatives. Only one dataset-loading function should be called in the `main()` method.

#### ALTERNATIVE #1 (MANUAL dataset upload)

This method allows users to browse their computers and select the dataset to import.

**EXAMPLE #2:**

```python
def load_dataset_man():
    # Manually upload the CSV using Google Colab.
    from google.colab import files
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No CSV file was uploaded.")

    print("\n**************************")
    print("*   Data Preprocessing   *")
    print("**************************\n")

    file_name = next(iter(uploaded))
    df = pd.read_csv(file_name, low_memory=False)
    print(f"Loaded {len(df):,} rows from {file_name}")
    return df

# In the main method:
raw_df = load_dataset_man()
# raw_df = load_dataset_auto()
```

**NOTE:** Make sure `raw_df = load_dataset_man()` is uncommented and the automatic loading call is commented out. The function uses Pandas (`pd`), which the notebook imports at the beginning.

1. Execute the notebook in Google Colab.
2. Click **Choose Files** when the upload window appears.
3. Locate and select **`processes_datasets.csv`** on your computer.
4. Wait for the program to process the dataset and display the results.

#### ALTERNATIVE #2 (AUTOMATIC dataset loading from Google Drive)

This method reads the dataset from a specified Google Drive directory without asking users to select a file each time.

**EXAMPLE #3:**

```python
def load_dataset_auto():
    # Automatically read the CSV from your Google Drive folder.
    from google.colab import drive
    drive.mount('/content/drive')
    file_path = '/content/drive/MyDrive/Adv OS Term Project/processes_datasets.csv'
    # Change file_path above if your Google Drive folder has a different name.

    print("\n**************************")
    print("*   Data Preprocessing   *")
    print("**************************\n")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df):,} rows from Google Drive")
    return df

# In the main method:
# raw_df = load_dataset_man()
raw_df = load_dataset_auto()
```

**NOTE:** The directory shown above is the directory used in the supplied program. If the dataset is stored in a different Google Drive folder, change `file_path` to its actual location. In the supplied notebook, `load_dataset_auto()` is inside a triple-quoted block; remove those triple quotes to enable it before choosing this alternative.

1. Upload **`processes_datasets.csv`** to the Google Drive folder you plan to use.
2. Confirm that the `file_path` in `load_dataset_auto()` matches the dataset's location.
3. Uncomment `raw_df = load_dataset_auto()` in `main()` and comment out `raw_df = load_dataset_man()`.
4. Execute the notebook.
5. Follow the Google Colab prompt to connect your Google Drive account.
6. Wait for the program to process the dataset and display the results.

**If you want to switch between alternatives:** Change which function is active in `main()` and rerun the notebook from the beginning. If an earlier session causes problems, use **Runtime → Disconnect and delete runtime** and start again.

## Expected Output of the Program

After successful execution, the program prints the dataset information, experiment settings, machine learning results, sample predictions, and CPU scheduling results. It then displays graphs and saves CSV files.

**EXAMPLE #4 (output format; numerical results are placeholders):**

```text
**************************
*   Data Preprocessing   *
**************************

Loaded [number] rows from [dataset file]
Valid records after cleaning: [number]
Selected 4,000 earliest valid jobs.
Training jobs: 3,200
Testing jobs: 800

***************************
*   Experiment Settings   *
***************************

Test size: 20%
NN batch size: 64
NN epochs: 25
RF trees: 100

Input attributes: ['SubmitTime', 'UserID', 'ReqTime', 'ReqMemory']
Target attribute: RunTime (seconds)

Training Random Forest...
Training Support Vector Regression...
Training Neural Network...

*******************************************
*   Machine Learning Regression Results   *
*******************************************

ML Model | MAE (seconds) | RMSE (seconds) | R² | Training time (s)
[Results for RF, SVR, and NN, sorted by MAE]

****************************************
*   Sample Test Jobs and Predictions   *
****************************************

[First 10 test jobs: JobID, SubmitTime, RunTime,
 RF_Predicted_RunTime, SVR_Predicted_RunTime, NN_Predicted_RunTime]

******************************
*   CPU Scheduling Results   *
******************************

Scheduler | Burst Times | Jobs | Avg Waiting Time (s) | Avg Turnaround Time (s)
FCFS      | Actual      | 800  | [value]              | [value]
FCFS      | RF          | 800  | [value]              | [value]
FCFS      | SVR         | 800  | [value]              | [value]
FCFS      | NN          | 800  | [value]              | [value]
SJF       | Actual      | 800  | [value]              | [value]
SJF       | RF          | 800  | [value]              | [value]
SJF       | SVR         | 800  | [value]              | [value]
SJF       | NN          | 800  | [value]              | [value]

[Five bar graphs are displayed]

Saved results as CSV files in the notebook's working directory.
```

**What do the results mean?**

- **Training time (s):** Time spent training each machine learning model, measured in seconds.
- **MAE:** Mean Absolute Error; the average absolute difference between predicted and actual runtimes. Lower values indicate smaller errors.
- **RMSE:** Root Mean Squared Error; a measure of prediction error that places greater weight on large errors. Lower values indicate smaller errors.
- **R²:** A score describing how well predicted runtimes account for variation in actual runtimes. Higher values generally indicate a better fit.
- **Average waiting time:** Average time a job waits before starting execution in the simulation.
- **Average turnaround time:** Average time between a job's arrival and completion in the simulation.

The program displays **five bar graphs**: one comparing ML prediction errors (MAE), two showing FCFS average waiting and turnaround times, and two showing the same measurements for SJF.

**NOTE:** In the predicted scenarios, the simulator uses predicted runtimes as the simulated execution durations. Therefore, shorter simulated waiting or turnaround times do not necessarily mean that real jobs would execute faster.

Record the results.

---

**END OF README DOCUMENT**
