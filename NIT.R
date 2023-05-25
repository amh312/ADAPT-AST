# This package provides the algorithms for prediction
# and prioritisation of antimicrobial susceptibility testing.
# Elements in CAPITALS are for substitution with the
# predictors and outcomes of choice.

#LEGEND
# amr_uti = amr_utiset = 
# NIT = Antibiotic for prediction (3 letters in capitals) = 
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
amr_uti$B1_NIT <-  amr_uti$`micro - prev resistance NIT ALL`
amr_uti$B2_NIT <-  amr_uti$`selected micro - colonization pressure NIT 90 - granular level`
amr_uti$B3_NIT <-  amr_uti$`medication 14 - nitrofurantoin`
amr_uti$B4_NIT <-  amr_uti$`custom 90 - nursing home`
amr_uti$B5_NIT <-  amr_uti$`hosp ward - OP`


#2.3 Split into training and testing amr_uti sets
tr_amr_uti <- amr_uti[amr_uti$is_train==1,] # 80% of amr_uti
te_amr_uti <- amr_uti[amr_uti$is_train==0,] # 20% of amr_uti

### MODULE 3: Model matrix generation ###

#3.1 Listing of variables


#3.2 Training predictor variable matrix
tr_x_NIT <- as.matrix(model.matrix( NIT ~
                                      B1_NIT + # Predictor variable 1
                                      B2_NIT +
                                      B3_NIT + # Predictor variable 1
                                      B4_NIT +
                                      B5_NIT , # Predictor variable 1 ,
                                    tr_amr_uti )) #training matrix formation

tr_x_NIT <- tr_x_NIT[,2:ncol(tr_x_NIT)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

#3.3 Testing predictor variable matrix
te_x_NIT <- as.matrix(model.matrix( NIT ~
                                      B1_NIT + # Predictor variable 1
                                      B2_NIT +
                                      B3_NIT + # Predictor variable 1
                                      B4_NIT +
                                      B5_NIT ,
                                    data = te_amr_uti )) #testing matrix formation

te_x_NIT <- te_x_NIT[,2:ncol(te_x_NIT)] # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame




### MODULE 4: Stan data list ###

NIT_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_NIT = nrow(tr_amr_uti) , # Training sample size
  n_te_NIT = nrow(te_amr_uti) , # Testing sample size
  x_n_NIT = ncol(tr_x_NIT) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_NIT = tr_x_NIT , # Predictor variable amr_uti (training)
  tr_y_NIT = tr_amr_uti$NIT , # Outcome variable amr_uti (training)
  te_x_NIT = te_x_NIT , # Predictor variable amr_uti (testing)
  
  # 4.3 Standard priors
  a1_m_NIT = 0 , a1_s_NIT = 10 , # Fixed intercept prior mean and sd
  B_m_NIT = 0 , B_s_NIT = 10 , # All-B prior mean and sd (if matrix used)
  B1_m_NIT = 0 , B1_s_NIT = 10 , # Fixed B1 prior mean and sd
  B2_m_NIT = 0 , B2_s_NIT = 10 , # Fixed B2 prior mean and sd
  B3_m_NIT = 0 , B3_s_NIT = 10 , # etc.
  B4_m_NIT = 0 , B4_s_NIT = 10 ,
  B5_m_NIT = 0 , B5_s_NIT = 10 ,
  B6_m_NIT = 0 , B6_s_NIT = 10 ,
  B7_m_NIT = 0 , B7_s_NIT = 10 ,
  B8_m_NIT = 0 , B8_s_NIT = 10 ,
  B9_m_NIT = 0 , B9_s_NIT = 10 ,
  B10_m_NIT = 0 , B10_s_NIT = 10 ,
  B11_m_NIT = 0 , B11_s_NIT = 10 ,
  B12_m_NIT = 0 , B12_s_NIT = 10 ,
  
  # 4.4 Random effects data inputs
  pt_n_NIT = length(unique(amr_uti$pt)) , # Number of group_idegories
  tr_pt_NIT = tr_amr_uti$pt , # category amr_uti (training)
  te_pt_NIT = te_amr_uti$pt , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_NIT = 0 ,  #Random intercept prior mean
  ar_sa_NIT = 0.5 , ar_sb_NIT = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_NIT = 0 ,  #Random slope prior mean
  br_sa_NIT = 0.5 , br_sb_NIT = 0.05 #Random slope sd prior alpha and beta
  
)




### MODULE 5: Stan MCMC algorithm ####

NIT_model = stan(
  file = "NIT_model.stan" ,
  data = NIT_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)




### MODULE 6: Model assessment

# 6.1 View sampler performance
stan_trace(NIT_model, pars = c ( 'B_NIT' ))

# 6.2 View model coefficients
plot(NIT_model, pars = c ( 'B_NIT'))

# 6.3 Match predictions with real outcomes

df_NIT_model <- as.data.frame( NIT_model )
df_NIT_model <- as.data.frame( list(
  # Summarise mean
  NIT_mean = apply(df_NIT_model , 2 , mean)))

df_NIT_model <- cbind(
  as.data.frame( list( NIT_actual = te_amr_uti$NIT ) ), 
  NIT_pred = df_NIT_model[ grep('NIT_y_pr' , rownames(df_NIT_model)) , ] )

# 6.4 Performance measures

# 6.4.1 Log Loss
LogLoss(df_NIT_model$NIT_pred ,
        df_NIT_model$NIT_actual )


# 6.4.2 ROC AUC
NIT_pred <- prediction( df_NIT_model$NIT_pred , df_NIT_model$NIT_actual )
NIT_perf <- performance( NIT_pred , "tpr" , "fpr" )
plot( NIT_perf , colorize=TRUE )
AUC( df_NIT_model$NIT_pred , df_NIT_model$NIT_actual )
