
# This package provides the algorithms for prediction
# and prioritisation of antimicrobial susceptibility testing.
# Elements in CAPITALS are for substitution with the
# predictors and outcomes of choice.

#LEGEND
# DATA = dataset = amr_uti 
# ABX = Antibiotic for prediction (3 letters in capitals) = CIP
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
  
  # Age / ethnicity category variable
  amr_uti$age_group <- amr_uti %>% group_by(`demographics - age`,
                                            `demographics - is_white`) %>%
    group_indices()
  
  # Composite previous CIP/LVX resistance variables
  amr_uti$`micro - prev resistance CIP CAT` <- amr_uti %>% group_by(`micro - prev resistance CIP ALL`,
                                                                    `micro - prev resistance CIP 180`,
                                                                    `micro - prev resistance CIP 90`,
                                                                    `micro - prev resistance CIP 30`,
                                                                    `micro - prev resistance CIP 14`) %>%
    group_indices()
  
  amr_uti$`micro - prev resistance LVX CAT` <- amr_uti %>% group_by(`micro - prev resistance LVX ALL`,
                                                                    `micro - prev resistance LVX 180`,
                                                                    `micro - prev resistance LVX 90`,
                                                                    `micro - prev resistance LVX 30`,
                                                                    `micro - prev resistance LVX 14`) %>%
    group_indices()
  
  #Composite nursing home variable
  amr_uti$`custom CAT - nursing home` <- amr_uti %>% group_by(`custom 90 - nursing home`,
                                                              `custom 30 - nursing home`,
                                                              `custom 14 - nursing home`,
                                                              `custom 7 - nursing home`) %>%
    group_indices()
  
  
  #Composite antimicrobial exposure variable
  amr_uti$`ab CAT - fluoroquinolone` <- amr_uti %>% group_by(`ab class 180 - fluoroquinolone`,
                                                             `medication 180 - ciprofloxacin`,
                                                             `ab class 90 - fluoroquinolone`,
                                                             `medication 90 - ciprofloxacin`,
                                                             `ab class 30 - fluoroquinolone`,
                                                             `medication 30 - ciprofloxacin`,
                                                             `ab class 14 - fluoroquinolone`,
                                                             `medication 14 - ciprofloxacin`) %>%
    group_indices()

  #Composite ward category variable
  amr_uti <-  mutate(amr_uti, `hosp ward - CAT` = case_when(`hosp ward - ER` == 1 ~ 1,
                                                         `hosp ward - ICU` == 1 ~ 2,
                                                         `hosp ward - IP` == 1 ~ 3,
                                                         `hosp ward - OP` == 1 ~ 4, TRUE ~ 0))
  
  
  #Composite time-weighted healthcare exposure quantifier
  #Sum healthcare exposure in each time period
  amr_uti <- amr_uti %>% rowwise() %>% mutate(
    `comorbidity 7` = sum(across(`comorbidity 7 - Arrhythmia`:`comorbidity 7 - HIV`))) %>%
    mutate(
      `comorbidity 14` = sum(across(`comorbidity 14 - Arrhythmia`:`comorbidity 14 - HIV`))) %>% 
    mutate(
      `comorbidity 30` = sum(across(`comorbidity 30 - Arrhythmia`:`comorbidity 30 - HIV`))) %>%
    mutate(
      `comorbidity 90` = sum(across(`comorbidity 90 - Arrhythmia`:`comorbidity 90 - HIV`))) %>%
    mutate(
      `comorbidity 180` = sum(across(`comorbidity 180 - Arrhythmia`:`procedure 180 - had parenteral nutrition`)))
  #Group by exposure quantity across the 4 time periods

amr_uti <- amr_uti %>% mutate(`comorbidity` = case_when(`comorbidity 180` == 0 ~ 0,
                                                        TRUE ~ 1))

# 2.2.2 Variable assignment
amr_uti$B1_CIP <-  amr_uti$`micro - prev resistance CIP CAT`
amr_uti$B2_CIP <-  amr_uti$`micro - prev resistance LVX CAT`
amr_uti$B3_CIP <-  amr_uti$`selected micro - colonization pressure CIP 90 - granular level`
amr_uti$B4_CIP <-  amr_uti$`ab CAT - fluoroquinolone`
amr_uti$B5_CIP <-  amr_uti$`custom CAT - nursing home`
amr_uti$B6_CIP <-  amr_uti$`hosp ward - CAT`


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
                                      B6_CIP,
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
                                      B6_CIP,
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
  age_group_n_CIP = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_CIP = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_CIP = te_amr_uti$age_group , # category amr_uti (testing)
  
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

# AUC = 0.6426022

