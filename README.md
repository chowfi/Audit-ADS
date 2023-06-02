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
All the input datasets can be linked using a unique string identifier, uid. uid identifies the date and location(latitude and longitude) for each sample. There are train labels, metadata, satellite data and elevation data. More information can be found in the 03_rds_final_report.pdf on this repo.

### Output

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


## Implementation and Validation

### Data cleaning / Pre-processing

In this section, we document several decision that the ADS winner made, which may have had consequences on both accuracy and fairness performance. Because the ADS does not use scikit-learn, we are not able to run ml-inspect and instead our inspection of the ADS is manual.

During pre-processing, the winner of this ADS did not normalize the elevation data and the satellite data already came normalized. One decision the winner of this ADS made was to create an ad-hoc cluster variable (represented by the ordinal variable ’cluster’) as there is substantial spatial variation in the target variable. For example, the south region had the lowest average severity levels at 1.57 while the west region was 3.74. By creating this ad-hoc variable, he was able to better model the patterns in the different regions based on more granular spatial groupings.

### High Level Implementation

The solution uses an ensemble of three different boosted tree models – XGBoost, CatBoost and LightBoost – and features such as region, date, location cluster, elevation and Sentinel-2 satellite images – red, blue and green spectral bands at 1,000 and 2,500 meters from latitude and longitude.

To optimize the tree models, the solution employed a process of hyper-tuning nine different parameters. Two of these parameters took integer values and the remaining seven were categorical. The integer-valued parameters were the number of boosted trees and maximum tree depth. The categorical parameters included the type of elevation data, type of XY coordinates, type of slope data, type of region data, a boolean indicating whether to use sample weights when fitting the model, a boolean indicating whether to treat categorical features as numeric, and the type of satellite data to use.

### ADS Validation

This ADS is validated by the region-averaged root mean squared error (RMSE). The smaller the error value, the more accurate the model is. Region-averaged RMSE is calculated by taking the square root of the mean of squared differences between estimated and observed severity values for each region and averaging across regions.

Given this performance metric, the author of the ADS used regression models to generate a continuous outcome. He then rounded the predictions to the nearest integer value. For the ensemble models, he averaged across the predictions before rounding, and then rounded the final result. The ADS met its goal of achieving high performance as evidenced by winning second place in the competition with a sufficiently low region-averaged RMSE of 0.7616. Using the sampled dataset, our implementation of the ADS scored a similar region-averaged RMSE on the test set of 0.7275.

## Outcomes

We study the performance of the ADS across four subpopulations. Since we are using a binary version of the outcome variable, we use metrics commonly studied in binary classification problems, which we outline below. To further assess the validity of our findings, we compare our results to the performance of the ADS across subpopulations using RMSE averaged over the four regions.

### Subpopulations

Based on the ACS data we have, we identify sensitive groups using the following construct definitions. We evaluate the performance of the ADS across these sensitive groups.

- **Above average poverty rate**: We have ACS data at both the census tract and state level. We create indicator variables that denote whether the poverty rate in a given census tract is above the statewide average.

- **Above average shares of racial and ethnic subgroups: We create a series of indicator variables, corresponding to racial and ethnic subgroups that denote whether a given census tract has an above average population share of each subgroup. We generate these indicator variables for each subgroup in comparison to the statewide average. We also create an indicator that aggregates all non-white racial subgroups. For this audit, we focus on the indicator for non-white racial subgroups to reduce the total number of subgroups that we discuss. However, an expansion of this audit should ultimately consider the performance of the ADS across all racial subgroups as there may be important heterogeneity.

- **Low Income Community: this is a designation from Internal Revenue Code §45D(e). Broadly, it refers to any census tract where the poverty rate is greater than or equal to 20 percent or the median family income is less than 80 percent of either the statewide median family income or the metropolitan area median family income–whichever is greater. We propose a modified version of this definition, given the complexity associated with recreating the first approach, that only compares the median family income to the statewide median.

