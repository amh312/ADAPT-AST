# This package provides the algorithms for prediction
# and prioritisation of antimicrobial susceptibility testing.
# Elements in CAPITALS are for substitution with the
# predictors and outcomes of choice.

#LEGEND
# amr_uti = amr_utiset = 
# CIP = Antibiotic for prediction (3 letters in capitals) = 
# group_id = group_idegory (for random effects) = 

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
library("amr_uti.table")
setwd( "/Users/alexhoward/Documents/Projects/PhD/AMR-UTI practice amr_utiset/R_logistic_regression" )
load( "~/Documents/Projects/PhD/AMR-UTI practice amr_utiset/R_logistic_regression/AMR_UTI.Ramr_uti" )




### MODULE 2: amr_uti set preparation ###

# 2.1 amr_uti upload
amr_uti <- read_csv( "amr_uti.csv" )


# 2.2 amr_uti cleaning

  # 2.2.1 Variable standardisation and formulation

  # Patient category variable creation
  amr_uti$pt <- rep(1, nrow(amr_uti))

  # 2.2.2 Variable assignment
  amr_uti$B1_CIP <-  amr_uti$`micro - prev resistance CIP ALL`
  amr_uti$B2_CIP <-  amr_uti$`micro - prev resistance LVX ALL`
  amr_uti$B3_CIP <-  amr_uti$`selected micro - colonization pressure CIP 90 - granular level`
  amr_uti$B4_CIP <-  amr_uti$`medication 14 - ciprofloxacin`
  amr_uti$B5_CIP <-  amr_uti$`custom 90 - nursing home`
  amr_uti$B6_CIP <-  amr_uti$`hosp ward - OP`

  
  #2.3 Split into training and testing amr_uti sets
  tr_amr_uti <- amr_uti[amr_uti$is_train==1,] # 80% of amr_uti
  te_amr_uti <- amr_uti[amr_uti$is_train==0,] # 20% of amr_uti
  
  ### MODULE 3: Model matrix generation ###
  
  #3.1 Listing of variables
  
  
  #3.2 Training predictor variable matrix
tr_x_CIP <- as.matrix(model.matrix( CIP ~
                                      B1_CIP + # Predictor variable 1
                                      B2_CIP +
                                      B3_CIP + # Predictor variable 1
                                      B4_CIP +
                                      B5_CIP + # Predictor variable 1
                                      B6_CIP ,
                                      tr_amr_uti )) #training matrix formation

tr_x_CIP <- tr_x_CIP[,2:ncol(tr_x_CIP)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

#3.3 Testing predictor variable matrix
te_x_CIP <- as.matrix(model.matrix( CIP ~
                                      B1_CIP + # Predictor variable 1
                                      B2_CIP +
                                      B3_CIP + # Predictor variable 1
                                      B4_CIP +
                                      B5_CIP + # Predictor variable 1
                                      B6_CIP ,
                                      data = te_amr_uti )) #testing matrix formation

te_x_CIP <- te_x_CIP[,2:ncol(te_x_CIP)] # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame




### MODULE 4: Stan data list ###

CIP_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_CIP = nrow(tr_amr_uti) , # Training sample size
  n_te_CIP = nrow(te_amr_uti) , # Testing sample size
  x_n_CIP = ncol(tr_x_CIP) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_CIP = tr_x_CIP , # Predictor variable amr_uti (training)
  tr_y_CIP = tr_amr_uti$CIP , # Outcome variable amr_uti (training)
  te_x_CIP = te_x_CIP , # Predictor variable amr_uti (testing)
  
  # 4.3 Standard priors
  a1_m_CIP = 0 , a1_s_CIP = 10 , # Fixed intercept prior mean and sd
  B_m_CIP = 0 , B_s_CIP = 10 , # All-B prior mean and sd (if matrix used)
  B1_m_CIP = 0 , B1_s_CIP = 10 , # Fixed B1 prior mean and sd
  B2_m_CIP = 0 , B2_s_CIP = 10 , # Fixed B2 prior mean and sd
  B3_m_CIP = 0 , B3_s_CIP = 10 , # etc.
  B4_m_CIP = 0 , B4_s_CIP = 10 ,
  B5_m_CIP = 0 , B5_s_CIP = 10 ,
  B6_m_CIP = 0 , B6_s_CIP = 10 ,
  B7_m_CIP = 0 , B7_s_CIP = 10 ,
  B8_m_CIP = 0 , B8_s_CIP = 10 ,
  B9_m_CIP = 0 , B9_s_CIP = 10 ,
  B10_m_CIP = 0 , B10_s_CIP = 10 ,
  B11_m_CIP = 0 , B11_s_CIP = 10 ,
  B12_m_CIP = 0 , B12_s_CIP = 10 ,
  
  # 4.4 Random effects data inputs
  pt_n_CIP = length(unique(amr_uti$pt)) , # Number of group_idegories
  tr_pt_CIP = tr_amr_uti$pt , # category amr_uti (training)
  te_pt_CIP = te_amr_uti$pt , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_CIP = 0 ,  #Random intercept prior mean
  ar_sa_CIP = 0.5 , ar_sb_CIP = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_CIP = 0 ,  #Random slope prior mean
  br_sa_CIP = 0.5 , br_sb_CIP = 0.05 #Random slope sd prior alpha and beta
  
)




### MODULE 5: Stan MCMC algorithm ####

CIP_model = stan(
  file = "CIP.stan" ,
  data = CIP_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)




### MODULE 6: Model assessment

# 6.1 View sampler performance
stan_trace(CIP_model, pars = c ( 'B_CIP' ))

# 6.2 View model coefficients
plot(CIP_model, pars = c ( 'B_CIP'))

# 6.3 Match predictions with real outcomes

df_CIP_model <- as.data.frame( CIP_model )
df_CIP_model <- as.data.frame( list(
  # Summarise mean
  CIP_mean = apply(df_CIP_model , 2 , mean)))

df_CIP_model <- cbind(
  as.data.frame( list( CIP_actual = te_amr_uti$CIP ) ), 
  CIP_pred = df_CIP_model[ grep('CIP_y_pr' , rownames(df_CIP_model)) , ] )

# 6.4 Performance measures

  # 6.4.1 Log Loss
  LogLoss(df_CIP_model$CIP_pred ,
        df_CIP_model$CIP_actual )


  # 6.4.2 ROC AUC
  CIP_pred <- prediction( df_CIP_model$CIP_pred , df_CIP_model$CIP_actual )
  CIP_perf <- performance( CIP_pred , "tpr" , "fpr" )
  plot( CIP_perf , colorize=TRUE )
  AUC( df_CIP_model$CIP_pred , df_CIP_model$CIP_actual )