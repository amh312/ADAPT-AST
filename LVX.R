# This package provides the algorithms for prediction
# and prioritisation of antimicrobial susceptibility testing.
# Elements in CAPITALS are for substitution with the
# predictors and outcomes of choice.

#LEGEND
# DATA = dataset = amr_uti 
# ABX = Antibiotic for prediction (3 letters in capitals) = LVX
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
amr_uti$B1_LVX <-  amr_uti$`micro - prev resistance LVX ALL`
amr_uti$B2_LVX <-  amr_uti$`micro - prev resistance LVX ALL`
amr_uti$B3_LVX <-  amr_uti$`selected micro - colonization pressure LVX 90 - granular level`
amr_uti$B4_LVX <-  amr_uti$`medication 14 - levofloxacin`
amr_uti$B5_LVX <-  amr_uti$`custom 90 - nursing home`
amr_uti$B6_LVX <-  amr_uti$`hosp ward - OP`


#2.3 Split into training and testing amr_uti sets
tr_amr_uti <- amr_uti[amr_uti$is_train==1,] # 80% of amr_uti
te_amr_uti <- amr_uti[amr_uti$is_train==0,] # 20% of amr_uti

### MODULE 3: Model matrix generation ###

#3.1 Listing of variables


#3.2 Training predictor variable matrix
tr_x_LVX <- as.matrix(model.matrix( LVX ~
                                      B1_LVX + # Predictor variable 1
                                      B2_LVX +
                                      B3_LVX + # Predictor variable 1
                                      B4_LVX +
                                      B5_LVX + # Predictor variable 1
                                      B6_LVX ,
                                    tr_amr_uti )) #training matrix formation

tr_x_LVX <- tr_x_LVX[,2:ncol(tr_x_LVX)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

#3.3 Testing predictor variable matrix
te_x_LVX <- as.matrix(model.matrix( LVX ~
                                      B1_LVX + # Predictor variable 1
                                      B2_LVX +
                                      B3_LVX + # Predictor variable 1
                                      B4_LVX +
                                      B5_LVX + # Predictor variable 1
                                      B6_LVX ,
                                    data = te_amr_uti )) #testing matrix formation

te_x_LVX <- te_x_LVX[,2:ncol(te_x_LVX)] # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame




### MODULE 4: Stan data list ###

LVX_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_LVX = nrow(tr_amr_uti) , # Training sample size
  n_te_LVX = nrow(te_amr_uti) , # Testing sample size
  x_n_LVX = ncol(tr_x_LVX) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_LVX = tr_x_LVX , # Predictor variable amr_uti (training)
  tr_y_LVX = tr_amr_uti$LVX , # Outcome variable amr_uti (training)
  te_x_LVX = te_x_LVX , # Predictor variable amr_uti (testing)
  
  # 4.3 Standard priors
  a1_m_LVX = 0 , a1_s_LVX = 10 , # Fixed intercept prior mean and sd
  B_m_LVX = 0 , B_s_LVX = 10 , # All-B prior mean and sd (if matrix used)
  B1_m_LVX = 0 , B1_s_LVX = 10 , # Fixed B1 prior mean and sd
  B2_m_LVX = 0 , B2_s_LVX = 10 , # Fixed B2 prior mean and sd
  B3_m_LVX = 0 , B3_s_LVX = 10 , # etc.
  B4_m_LVX = 0 , B4_s_LVX = 10 ,
  B5_m_LVX = 0 , B5_s_LVX = 10 ,
  B6_m_LVX = 0 , B6_s_LVX = 10 ,
  B7_m_LVX = 0 , B7_s_LVX = 10 ,
  B8_m_LVX = 0 , B8_s_LVX = 10 ,
  B9_m_LVX = 0 , B9_s_LVX = 10 ,
  B10_m_LVX = 0 , B10_s_LVX = 10 ,
  B11_m_LVX = 0 , B11_s_LVX = 10 ,
  B12_m_LVX = 0 , B12_s_LVX = 10 ,
  
  # 4.4 Random effects data inputs
  age_group_n_LVX = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_LVX = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_LVX = te_amr_uti$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_LVX = 0 ,  #Random intercept prior mean
  ar_sa_LVX = 0.5 , ar_sb_LVX = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_LVX = 0 ,  #Random slope prior mean
  br_sa_LVX = 0.5 , br_sb_LVX = 0.05 #Random slope sd prior alpha and beta
  
)




### MODULE 5: Stan MCMC algorithm ####

LVX_model = stan(
  file = "LVX.stan" ,
  data = LVX_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)




### MODULE 6: Model assessment

# 6.1 View sampler performance
stan_trace(LVX_model, pars = c ( 'B_LVX' ))

# 6.2 View model coefficients
plot(LVX_model, pars = c ( 'B_LVX'))

# 6.3 Match predictions with real outcomes

df_LVX_model <- as.data.frame( LVX_model )
df_LVX_model <- as.data.frame( list(
  # Summarise mean
  LVX_mean = apply(df_LVX_model , 2 , mean)))

df_LVX_model <- cbind(
  as.data.frame( list( LVX_actual = te_amr_uti$LVX ) ), 
  LVX_pred = df_LVX_model[ grep('LVX_y_pr' , rownames(df_LVX_model)) , ] )

# 6.4 Performance measures

# 6.4.1 Log Loss
LogLoss(df_LVX_model$LVX_pred ,
        df_LVX_model$LVX_actual )


# 6.4.2 ROC AUC
LVX_pred <- prediction( df_LVX_model$LVX_pred , df_LVX_model$LVX_actual )
LVX_perf <- performance( LVX_pred , "tpr" , "fpr" )
plot( LVX_perf , colorize=TRUE )
AUC( df_LVX_model$LVX_pred , df_LVX_model$LVX_actual )

# AUC = 0.631306
