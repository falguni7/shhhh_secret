# Proposed Method

## Installation
This code is based on `Python 3.6`, all requirements are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```

## Dataset

### AIOps (KPI, IOpsCompetition) and UCR. 
1. AIOps Link: https://github.com/NetManAIOps/KPI-Anomaly-Detection
2. UCR Link: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ 
and https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
3. Download and unzip the data in `data/iops_competition` and `data/ucr` respectively. 
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.

### SWaT and WADI. 
1. For SWaT and WADI, you need to apply by their official tutorial. Link: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
2. Because multiple versions of these two datasets exist, 
we used their newer versions: `SWaT.A1 & A2_Dec2015` and `WADI.A2_19Nov2019`.
3. Download and unzip the data in `data/swat` and `data/wadi` respectively. Then run the 
`swat_preprocessing()` and `wadi_preprocessing()` functions in `dataloader/data_preprocessing.py` for preprocessing.


## Command to run proposed method
```
# dataset_name: IOpsCompetition, UCR, SWaT, WADI)
python cutAddPaste.py --selected_dataset <dataset_name> --device cuda --seed 2
```

