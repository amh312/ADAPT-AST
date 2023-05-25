# This package provides the algorithms for prediction
# and prioritisation of antimicrobial susceptibility testing.
# Elements in CAPITALS are for substitution with the
# predictors and outcomes of choice.

#LEGEND
# DATA = dataset = amr_uti 
# ABX = Antibiotic for prediction (3 letters in capitals) = SXT
# CAT = Category (for random effects) = pt

#MODEL TITLE: 

### MODULE 1: Package loading

library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library("tidyverse")
library("MLmetrics")
library("ROCR")
library("mlmRev")
library("lme4")
library("rstanarm")
setwd("/Users/alexhoward/Documents/Projects/PhD/ADAPT-AST")
load("~/Documents/Projects/PhD/ADAPT-AST/ABX.RData")




### MODULE 2: amr_uti set preparation ###

# 2.1 amr_uti upload
amr_uti <- read_csv( "amr_uti.csv" )


# 2.2 amr_uti cleaning

# 2.2.1 Variable standardisation and formulation

# Age group category variable creation
amr_uti$age_group <- amr_uti %>% group_by(`demographics - age`,
                                          `demographics - is_white`) %>%
  group_indices()

# 2.2.2 Variable assignment
amr_uti$B1_SXT <-  amr_uti$`micro - prev resistance SXT ALL`
amr_uti$B2_SXT <-  amr_uti$`selected micro - colonization pressure SXT 90 - granular level`
amr_uti$B3_SXT <-  amr_uti$`medication 14 - trimethoprim/sulfamethoxazole`
amr_uti$B4_SXT <-  amr_uti$`custom 90 - nursing home`
amr_uti$B5_SXT <-  amr_uti$`hosp ward - OP`


#2.3 Split into training and testing amr_uti sets
tr_amr_uti <- amr_uti[amr_uti$is_train==1,] # 80% of amr_uti
te_amr_uti <- amr_uti[amr_uti$is_train==0,] # 20% of amr_uti

### MODULE 3: Model matrix generation ###

#3.1 Listing of variables


#3.2 Training predictor variable matrix
tr_x_SXT <- as.matrix(model.matrix( SXT ~
                                      B1_SXT + # Predictor variable 1
                                      B2_SXT +
                                      B3_SXT + # Predictor variable 1
                                      B4_SXT +
                                      B5_SXT , # Predictor variable 1 ,
                                    tr_amr_uti )) #training matrix formation

tr_x_SXT <- tr_x_SXT[,2:ncol(tr_x_SXT)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

#3.3 Testing predictor variable matrix
te_x_SXT <- as.matrix(model.matrix( SXT ~
                                      B1_SXT + # Predictor variable 1
                                      B2_SXT +
                                      B3_SXT + # Predictor variable 1
                                      B4_SXT +
                                      B5_SXT ,
                                    data = te_amr_uti )) #testing matrix formation

te_x_SXT <- te_x_SXT[,2:ncol(te_x_SXT)] # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame




### MODULE 4: Stan data list ###

SXT_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_SXT = nrow(tr_amr_uti) , # Training sample size
  n_te_SXT = nrow(te_amr_uti) , # Testing sample size
  x_n_SXT = ncol(tr_x_SXT) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_SXT = tr_x_SXT , # Predictor variable amr_uti (training)
  tr_y_SXT = tr_amr_uti$SXT , # Outcome variable amr_uti (training)
  te_x_SXT = te_x_SXT , # Predictor variable amr_uti (testing)
  
  # 4.3 Standard priors
  a1_m_SXT = 0 , a1_s_SXT = 10 , # Fixed intercept prior mean and sd
  B_m_SXT = 0 , B_s_SXT = 10 , # All-B prior mean and sd (if matrix used)
  B1_m_SXT = 0 , B1_s_SXT = 10 , # Fixed B1 prior mean and sd
  B2_m_SXT = 0 , B2_s_SXT = 10 , # Fixed B2 prior mean and sd
  B3_m_SXT = 0 , B3_s_SXT = 10 , # etc.
  B4_m_SXT = 0 , B4_s_SXT = 10 ,
  B5_m_SXT = 0 , B5_s_SXT = 10 ,
  B6_m_SXT = 0 , B6_s_SXT = 10 ,
  B7_m_SXT = 0 , B7_s_SXT = 10 ,
  B8_m_SXT = 0 , B8_s_SXT = 10 ,
  B9_m_SXT = 0 , B9_s_SXT = 10 ,
  B10_m_SXT = 0 , B10_s_SXT = 10 ,
  B11_m_SXT = 0 , B11_s_SXT = 10 ,
  B12_m_SXT = 0 , B12_s_SXT = 10 ,
  
  # 4.4 Random effects data inputs
  age_group_n_SXT = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_SXT = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_SXT = te_amr_uti$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_SXT = 0 ,  #Random intercept prior mean
  ar_sa_SXT = 0.5 , ar_sb_SXT = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_SXT = 0 ,  #Random slope prior mean
  br_sa_SXT = 0.5 , br_sb_SXT = 0.05 #Random slope sd prior alpha and beta
  
)




### MODULE 5: Stan MCMC algorithm ####

SXT_model = stan(
  file = "SXT.stan" ,
  data = SXT_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)




### MODULE 6: Model assessment

# 6.1 View sampler performance
stan_trace(SXT_model, pars = c ( 'B_SXT' ))

# 6.2 View model coefficients
plot(SXT_model, pars = c ( 'B_SXT'))

# 6.3 Match predictions with real outcomes

df_SXT_model <- as.data.frame( SXT_model )
df_SXT_model <- as.data.frame( list(
  # Summarise mean
  SXT_mean = apply(df_SXT_model , 2 , mean)))

df_SXT_model <- cbind(
  as.data.frame( list( SXT_actual = te_amr_uti$SXT ) ), 
  SXT_pred = df_SXT_model[ grep('SXT_y_pr' , rownames(df_SXT_model)) , ] )

# 6.4 Performance measures

# 6.4.1 Log Loss
LogLoss(df_SXT_model$SXT_pred ,
        df_SXT_model$SXT_actual )


# 6.4.2 ROC AUC
SXT_pred <- prediction( df_SXT_model$SXT_pred , df_SXT_model$SXT_actual )
SXT_perf <- performance( SXT_pred , "tpr" , "fpr" )
plot( SXT_perf , colorize=TRUE )
AUC( df_SXT_model$SXT_pred , df_SXT_model$SXT_actual )

#AUC = 0.5791894