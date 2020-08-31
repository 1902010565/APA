ReadMe Leveraging PDPs and ICEs to Identify Heterogeneity in Conditional Average Treatment Effects

Student Numbers 601631 & 534751

Humboldt University APA Course 2020 with Prof. Stefan Lessmann


To replicate figures for generated data, see the functions section at the top of the Figure Replication file and then find the section of the figure to be replicated and the code.

Generated data will consist of 5000 observations, 10 covariates (minimum), random treatment selection and a variance of 2 for randomly generated figures. A treatment effect must be specified by assigning theta_num from 1 to 8. The patterns are as follows:

1 = theta sine with other vars (reducing in influence from v1 to v9) but lin with v10 only above 0,
2 = theta sine with other vars (reducing in influence from v1 to v9) but lin in v10,
3 = theta sine with other vars (reducing in influence from v1 to v9) but neg linear in v10 with a lot of noise
4 = theta sine with other vars (reducing in influence from v1 to v8) but linear in v10 with interaction v8
5 = theta sine with other vars (reducing in influence from v1 to v8) but linear in v10 with switch interaction in v9
6 = theta sine with other vars (reducing in influence from v1 to v9) but absolute with v10
7 = theta sine with other vars (reducing in influence from v1 to v9) but absolute with v10 with more noise
8 = theta sine with other vars (reducing in influence from v1 to v8) but heavy underlying interaction between v10 and v9

The variable of interest is always variable 10 (V10). The dummy variable is always variable 9 (V9) and randomly assigned.

Treatment assignment can be adjusted by selecting “imbalanced” which only assigns treatment to 20% of data points, “linear” where the chance of treatment is linearly correlated with variables.

The sampleCATEestimation function uses a Ranger T learner to perform a CATE estimation. The train-test split can be adjusted by changing test_split from its default of 0.8 which is 80% for training, 20% for testing. The outputs of this function is a list which includes the dataframe ready for PDP regression, MSE and the approximated CATE.

The xgboost_cate function performs xgboost to estimate CATE, based on the dataframe from the data generation function. It returns a dataframe of 10 variables and a treatment effect variable, which is later used to generate ICE plots. The proportion between train and test split is 80 to 20.

Then, run the block of code which corresponds to the PDPs or ICEs to be created.

To replicate figures for the real dataset, see the Jupyter Notebook file.