# Audit_ADS
This audit of the automated decision system is a group project developed by a team of two individuals.

The ADS uses satellite imagery to classify the severity of cyanobacteria blooms in small water bodies. The data and ADS come from a data science competition hosted by NASA and DrivenData. Algal blooms are an environmental justice issue. Because this ADS could help water quality managers better allocate resources which influence public health warnings for drinking water and recreation, the algorithm must be accurate not only for the overall population but also for sensitive subpopulations.

Relevant Links <br>
• Original Second Place Repository: https://github.com/apwheele/algaebloom <br>
• Competition Website: https://www.drivendata.org/competitions/143/tick-tick-bloom/page/650/ <br>

## The repo consists of:
01_train_test_split.ipynb <br>
02_rds_report_final.ipynb <br>
03_rds_final_report.pdf <br>
04_presentation_slides.pdf <br>
05_metadata.csv <br>
06_train_labels.csv <br>
07_test_labels.csv <br>
08_algae_pts_with_tracts.csv <br>

## Background

The purpose of this automated decision-making system (ADS) is to use satellite imagery to detect and classify the severity of cyanobacteria blooms in small, inland water bodies. This will help water quality managers better allocate resources for in-situ sampling and make more informed decisions around public health warnings for drinking water and recreation. The primary goal of this ADS is to achieve accuracy. The data and ADS come from a data science competition hosted by NASA and DrivenData. DrivenData is a company that hosts social impact data science competitions like those found on Kaggle. We chose to audit the solution of the second-placed winner for the "Tick Tick Bloom: Harmful Algal Bloom Detection Challenge" that completed on Feb 17, 2023. The data and code implementing the ADS can be found on the winner’s GitHub page with documentation. DrivenData requires that all winning solutions are open source under The MIT License.

We selected this particular ADS because we were curious to see if there were subpopulations for which the ADS was less accurate in prediction. Algal blooms are an environmental justice issue. Exposure to high levels of cyanobacteria has been linked to cancer, birth defects, and even death (Gorham et al., 2020; Schaider et al., 2019). Prior research suggests that in the U.S. there are significant racial and socioeconomic disparities in access to clean drinking water (Schaider et al., 2019). Because this ADS could help water quality managers better allocate resources for in-situ sampling and make informed decisions around public health warnings for drinking water and recreation, the algorithm must be accurate not only for the overall population but also for sensitive subpopulations.

## Input and Output

### Description

The competition provides 23,570 rows of training and test data. Since the test data do not have labels, we conduct our audit using the training data (n=17,060). Each row in the training data is a unique in-situ sample collected for a given date and location going back to 2013. We sample from the training data to speed up the runtime of training and hyperparameter tuning. First, we restrict our data to instances recorded after 2016. From the filtered data, we sample so as to ensure proportional levels in each region to the original data while using a 72%/28% train/test split that is similar to what was done during the competition. To illustrate this, we present the number of sampled training observations by year in figure 1. Since the competition test data used samples with unseen latitude and longitude (i.e., "spatial holdouts"), we made sure there was no spatial overlap in our training and test data. Also similar to the ADS implementation, we then split the new training data (the 72%) further into a training and validation set. We tune the hyperparameters on the validation set using the same approach to tuning as in the competition. The author of the ADS uses custom functions for hyperparameter tuning. The function takes in a defined number of rows for selecting the validation set and performs cross-validation across 10 splits. We report performance on the test set that was unseen during training and validation.

Auxiliary Data. We are using American Community Survey (ACS) 2015-2019 5-year estimates at the census tract level for our audit. We chose to use the 2015-2019 estimates, as there have been documented problems and delays associated with subsequent surveys due to COVID-19 (Wines and Cramer). There are approximately 74,000 census tracts for all 50 U.S. states, Washington D.C., and Puerto Rico. However, only 1,660 census tracts are uniquely represented in the training data. This occurs because some in-situ samples appear to come clustered in the same geographic areas. For example, we observed that 5,308 samples come from only a handful of census tracts in Chatham County, North Carolina. The features we have collected using census data include: race, ethnicity, median household income, and poverty rate.

### Input
All the input datasets can be linked using a unique string identifier, uidg. uid identifies the date and location(latitude and longitude) for each sample. There are train labels, metadata, satellite data and elevation data. More information can be found in the 03_rds_final_report.pdf on this repo.

### Output

Markdown for a GitHub README would look the same as the conversion I previously made. Below is the same text re-formatted into Markdown, suitable for a GitHub README:

---

## Output

The output of the system formally is a severity level that takes integer values 1 through 5. According to DrivenData, the severity is based on cyanobacteria density. Cyanobacteria density is another column in the training data, which ranges from 0 to 804,667,500 cells per mL. 

As Table 5 illustrates, density values are non-overlapping for distinct severity levels and higher density values are associated with higher severity levels. For example, severity level 5 encompasses the highest density values, greater than 10 million cells per mL. According to the World Health Organization (WHO), moderate and high risk health exposures occur at density levels ≥ 20, 000 cells per mL (WHO, 2003). 

Using this classification approach, we construct a binary outcome label for a high risk health exposure. The binarized label aligns with severity levels 2-5, as Table A1 indicates. Slightly less than half (≈ 44 percent) of the training data are low-risk. DrivenData specifies that the submission is based on severity levels and not the raw, underlying densities.

**Table 5: Formal severity level ranges**

| Severity | Density range (cells/mL) |
|:--------:|:------------------------:|
|    1     | < 20, 000                |
|    2     | 20, 000− < 100, 000      |
|    3     | 100, 000− < 1, 000, 000  |
|    4     | 1, 000, 000− < 10, 000,000|
|    5     | ≥ 10, 000, 000           |

**Table 6: Binarized severity based on estimated risk**

| Severity | Percent of Total |
|:--------:|:----------------:|
|    Low   |        38        |
|    High  |                  |
|    4     |        22        |
|    2     |        20        |
|    3     |        20        |
|    5     |        < 1       |

---


