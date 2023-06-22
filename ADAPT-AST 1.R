
library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library("tidyverse")
library("MLmetrics")
library("ROCR")
library("mlmRev")
library("lme4")
library("rstanarm")
library("DescTools")
setwd("/Users/alexhoward/Documents/Projects/PhD/ADAPT-AST")
load("~/Documents/Projects/PhD/ADAPT-AST/ABX.RData")





# 1. VARIABLE ASSIGMENT

##SXT
amr_uti$B1_SXT <-  amr_uti$`micro - prev resistance SXT ALL`
amr_uti$B2_SXT <-  amr_uti$`selected micro - colonization pressure SXT 90 - granular level`
amr_uti$B3_SXT <-  amr_uti$`medication 14 - trimethoprim/sulfamethoxazole`
amr_uti$B4_SXT <-  amr_uti$`custom 90 - nursing home`
amr_uti$B5_SXT <-  amr_uti$`hosp ward - OP`

tr_amr_uti <- amr_uti[amr_uti$is_train==1,]
te_amr_uti <- amr_uti[amr_uti$is_train==0,]

tr_x_SXT <- as.matrix(model.matrix( SXT ~
                                      B1_SXT +
                                      B2_SXT +
                                      B3_SXT +
                                      B4_SXT +
                                      B5_SXT ,
                                    tr_amr_uti ))

tr_x_SXT <- tr_x_SXT[,2:ncol(tr_x_SXT)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

te_x_SXT <- as.matrix(model.matrix( SXT ~
                                      B1_SXT +
                                      B2_SXT +
                                      B3_SXT + 
                                      B4_SXT +
                                      B5_SXT ,
                                    data = te_amr_uti ))

te_x_SXT <- te_x_SXT[,2:ncol(te_x_SXT)] # Intercept removal
te_amr_uti <- as.data.frame(te_amr_uti) # Convert back to data frame

#NIT
amr_uti$B1_NIT <-  amr_uti$`micro - prev resistance NIT ALL`
amr_uti$B2_NIT <-  amr_uti$`selected micro - colonization pressure NIT 90 - granular level`
amr_uti$B3_NIT <-  amr_uti$`medication 14 - nitrofurantoin`
amr_uti$B4_NIT <-  amr_uti$`custom 90 - nursing home`
amr_uti$B5_NIT <-  amr_uti$`hosp ward - OP`

tr_amr_uti <- amr_uti[amr_uti$is_train==1,]
te_amr_uti <- amr_uti[amr_uti$is_train==0,]

tr_x_NIT <- as.matrix(model.matrix( NIT ~
                                      B1_NIT +
                                      B2_NIT +
                                      B3_NIT +
                                      B4_NIT +
                                      B5_NIT ,
                                    tr_amr_uti ))

tr_x_NIT <- tr_x_NIT[,2:ncol(tr_x_NIT)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

te_x_NIT <- as.matrix(model.matrix( NIT ~
                                      B1_NIT +
                                      B2_NIT +
                                      B3_NIT +
                                      B4_NIT +
                                      B5_NIT ,
                                    te_amr_uti ))

te_x_NIT <- te_x_NIT[,2:ncol(te_x_NIT)]     # Intercept removal
te_amr_uti <- as.data.frame(te_amr_uti) # Convert back to data frame

##LVX
amr_uti$B1_LVX <-  amr_uti$`micro - prev resistance LVX ALL`
amr_uti$B2_LVX <-  amr_uti$`micro - prev resistance CIP ALL`
amr_uti$B3_LVX <-  amr_uti$`selected micro - colonization pressure LVX 90 - granular level`
amr_uti$B4_LVX <-  amr_uti$`medication 14 - levofloxacin`
amr_uti$B5_LVX <-  amr_uti$`custom 90 - nursing home`
amr_uti$B6_LVX <-  amr_uti$`hosp ward - OP`

tr_amr_uti <- amr_uti[amr_uti$is_train==1,]
te_amr_uti <- amr_uti[amr_uti$is_train==0,]

tr_x_LVX <- as.matrix(model.matrix( LVX ~
                                      B1_LVX +
                                      B2_LVX +
                                      B3_LVX +
                                      B4_LVX +
                                      B5_LVX +
                                      B6_LVX ,
                                    tr_amr_uti ))

tr_x_LVX <- tr_x_LVX[,2:ncol(tr_x_LVX)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

te_x_LVX <- as.matrix(model.matrix( LVX ~
                                      B1_LVX +
                                      B2_LVX +
                                      B3_LVX +
                                      B4_LVX +
                                      B5_LVX +
                                      B6_LVX ,
                                    data = te_amr_uti )) #testing matrix formation

te_x_LVX <- te_x_LVX[,2:ncol(te_x_LVX)] # Intercept removal
te_amr_uti <- as.data.frame(te_amr_uti) # Convert back to data frame

##CIP
amr_uti$B1_CIP <-  amr_uti$`micro - prev resistance CIP ALL`
amr_uti$B2_CIP <-  amr_uti$`micro - prev resistance LVX ALL`
amr_uti$B3_CIP <-  amr_uti$`selected micro - colonization pressure CIP 90 - granular level`
amr_uti$B4_CIP <-  amr_uti$`ab class ALL - fluoroquinolone`
amr_uti$B5_CIP <-  amr_uti$`custom 90 - nursing home`
amr_uti$B6_CIP <-  amr_uti$`hosp ward - OP`
amr_uti$fluo
tr_amr_uti <- amr_uti[amr_uti$is_train==1,]
te_amr_uti <- amr_uti[amr_uti$is_train==0,]

tr_x_CIP <- as.matrix(model.matrix( CIP ~
                                      B1_CIP +
                                      B2_CIP +
                                      B3_CIP +
                                      B4_CIP +
                                      B5_CIP +
                                      B6_CIP,
                                    tr_amr_uti ))

tr_x_CIP <- tr_x_CIP[,2:ncol(tr_x_CIP)]     # Intercept removal
tr_amr_uti <- as.data.frame(tr_amr_uti) # Convert back to data frame

te_x_CIP <- as.matrix(model.matrix( CIP ~
                                      B1_CIP + # Predictor variable 1
                                      B2_CIP +
                                      B3_CIP + # Predictor variable 1
                                      B4_CIP +
                                      B5_CIP + # Predictor variable 1
                                      B6_CIP,
                                    data = te_amr_uti )) #testing matrix formation

te_x_CIP <- te_x_CIP[,2:ncol(te_x_CIP)]
te_amr_uti <- as.data.frame(te_amr_uti)







# 2. STAN DATA INPUTS

##SXT
SXT_stan_amr_uti <- list(
  n_tr_SXT = nrow(tr_amr_uti) ,
  n_te_SXT = nrow(te_amr_uti) , # Testing sample size
  x_n_SXT = ncol(tr_x_SXT) , # Number of beta coefficients
  tr_x_SXT = tr_x_SXT , # Predictor variable amr_uti (training)
  tr_y_SXT = tr_amr_uti$SXT , # Outcome variable amr_uti (training)
  te_x_SXT = te_x_SXT , # Predictor variable amr_uti (testing)
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
  age_group_n_SXT = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_SXT = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_SXT = te_amr_uti$age_group , # category amr_uti (testing)
  ar_m_SXT = 0 ,  #Random intercept prior mean
  ar_sa_SXT = 0.5 , ar_sb_SXT = 0.05 , #Random intercept sd prior alpha and beta
  br_m_SXT = 0 ,  #Random slope prior mean
  br_sa_SXT = 0.5 , br_sb_SXT = 0.05 #Random slope sd prior alpha and beta
)

##NIT
NIT_stan_amr_uti <- list(
  n_tr_NIT = nrow(tr_amr_uti) , # Training sample size
  n_te_NIT = nrow(te_amr_uti) , # Testing sample size
  x_n_NIT = ncol(tr_x_NIT) , # Number of beta coefficients
  tr_x_NIT = tr_x_NIT , # Predictor variable amr_uti (training)
  tr_y_NIT = tr_amr_uti$NIT , # Outcome variable amr_uti (training)
  te_x_NIT = te_x_NIT , # Predictor variable amr_uti (testing)
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
  age_group_n_NIT = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_NIT = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_NIT = te_amr_uti$age_group , # category amr_uti (testing)
  ar_m_NIT = 0 ,  #Random intercept prior mean
  ar_sa_NIT = 0.5 , ar_sb_NIT = 0.05 , #Random intercept sd prior alpha and beta
  br_m_NIT = 0 ,  #Random slope prior mean
  br_sa_NIT = 0.5 , br_sb_NIT = 0.05 #Random slope sd prior alpha and beta
)

##LVX
LVX_stan_amr_uti <- list(
  n_tr_LVX = nrow(tr_amr_uti) , # Training sample size
  n_te_LVX = nrow(te_amr_uti) , # Testing sample size
  x_n_LVX = ncol(tr_x_LVX) , # Number of beta coefficients
  tr_x_LVX = tr_x_LVX , # Predictor variable amr_uti (training)
  tr_y_LVX = tr_amr_uti$LVX , # Outcome variable amr_uti (training)
  te_x_LVX = te_x_LVX , # Predictor variable amr_uti (testing)
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
  age_group_n_LVX = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_LVX = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_LVX = te_amr_uti$age_group , # category amr_uti (testing)
  ar_m_LVX = 0 ,  #Random intercept prior mean
  ar_sa_LVX = 0.5 , ar_sb_LVX = 0.05 , #Random intercept sd prior alpha and beta
  br_m_LVX = 0 ,  #Random slope prior mean
  br_sa_LVX = 0.5 , br_sb_LVX = 0.05 #Random slope sd prior alpha and beta
)

##CIP
CIP_stan_amr_uti <- list(
  n_tr_CIP = nrow(tr_amr_uti) , # Training sample size
  n_te_CIP = nrow(te_amr_uti) , # Testing sample size
  x_n_CIP = ncol(tr_x_CIP) , # Number of beta coefficients
  tr_x_CIP = tr_x_CIP , # Predictor variable amr_uti (training)
  tr_y_CIP = tr_amr_uti$CIP , # Outcome variable amr_uti (training)
  te_x_CIP = te_x_CIP , # Predictor variable amr_uti (testing)
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
  age_group_n_CIP = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_CIP = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_CIP = te_amr_uti$age_group , # category amr_uti (testing)
  ar_m_CIP = 0 ,  #Random intercept prior mean
  ar_sa_CIP = 0.5 , ar_sb_CIP = 0.05 , #Random intercept sd prior alpha and beta
  br_m_CIP = 0 ,  #Random slope prior mean
  br_sa_CIP = 0.5 , br_sb_CIP = 0.05 #Random slope sd prior alpha and beta
)







# 3. STAN MCMC ALGORITHMS

##SXT
SXT_model = stan(
  file = "SXT.stan" ,
  data = SXT_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)

df_SXT_model <- as.data.frame( SXT_model )
df_SXT_model <- as.data.frame( list(
  SXT_mean = apply(df_SXT_model , 2 , mean)))

df_SXT_model <- cbind(
  as.data.frame( list( SXT_actual = te_amr_uti$SXT ) ), 
  SXT_pred = df_SXT_model[ grep('SXT_y_pr' , rownames(df_SXT_model)) , ] )


##NIT
NIT_model = stan(
  file = "NIT.stan" ,
  data = NIT_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)

df_NIT_model <- as.data.frame( NIT_model )
df_NIT_model <- as.data.frame( list(
  NIT_mean = apply(df_NIT_model , 2 , mean)))

df_NIT_model <- cbind(
  as.data.frame( list( NIT_actual = te_amr_uti$NIT ) ), 
  NIT_pred = df_NIT_model[ grep('NIT_y_pr' , rownames(df_NIT_model)) , ] )

##LVX
LVX_model = stan(
  file = "LVX.stan" ,
  data = LVX_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)

df_LVX_model <- as.data.frame( LVX_model )
df_LVX_model <- as.data.frame( list(
  LVX_mean = apply(df_LVX_model , 2 , mean)))

df_LVX_model <- cbind(
  as.data.frame( list( LVX_actual = te_amr_uti$LVX ) ), 
  LVX_pred = df_LVX_model[ grep('LVX_y_pr' , rownames(df_LVX_model)) , ] )

##CIP
CIP_model = stan(
  file = "CIP.stan" ,
  data = CIP_stan_amr_uti ,
  iter = 1000 ,
  warmup = 500 ,
  chains = 3 ,
  cores = 3
)

df_CIP_model <- as.data.frame( CIP_model )
df_CIP_model <- as.data.frame( list(
  CIP_mean = apply(df_CIP_model , 2 , mean)))

df_CIP_model <- cbind(
  as.data.frame( list( CIP_actual = te_amr_uti$CIP ) ), 
  CIP_pred = df_CIP_model[ grep('CIP_y_pr' , rownames(df_CIP_model)) , ] )








## 4. TEST PRIORITISATION ALGORITHM

#Combine predicted probabilities
df_ABX_models <- cbind(SXT_rank = round(df_SXT_model$SXT_pred, 2),
                       NIT_rank = round(df_NIT_model$NIT_pred, 2),
                       LVX_rank = round(df_LVX_model$LVX_pred, 2),
                       CIP_rank = round(df_CIP_model$CIP_pred, 2))

#Calculate distances from 0.5
df_ABX_models <- abs(0.5-df_ABX_models)

#Rank by distances from 0.5
ABX_rank <- as.data.frame(t(apply(df_ABX_models, 1, rank, ties.method = "first")))

#Rename predicted probabilities columns
df_ABX_models <- cbind(SXT_prob = round(df_SXT_model$SXT_pred, 2),
                       NIT_prob = round(df_NIT_model$NIT_pred, 2),
                       LVX_prob = round(df_LVX_model$LVX_pred, 2),
                       CIP_prob = round(df_CIP_model$CIP_pred, 2))

# Return recommendations
ABX_rank <- ABX_rank %>% rowwise %>% mutate(
  Recommend_1 = case_when( SXT_rank == 1 ~ "SXT",
             NIT_rank == 1 ~ "NIT",
             LVX_rank == 1 ~ "LVX",
             TRUE ~ "CIP"),
  Recommend_2 = case_when( SXT_rank == 2 ~ "SXT",
                           NIT_rank == 2 ~ "NIT",
                           LVX_rank == 2 ~ "LVX",
                           TRUE ~ "CIP"))

#Create data frame of probs, ranks and actual results
ABX_rank <- cbind(df_ABX_models,
                  ABX_rank,
                  SXT_real = df_SXT_model$SXT_actual,
                  NIT_real = df_NIT_model$NIT_actual,
                  LVX_real = df_LVX_model$LVX_actual,
                  CIP_real = df_CIP_model$CIP_actual)

#Create test result returns
ABX_rank <- ABX_rank %>% rowwise %>% mutate(
  ADAPT_SXT = case_when(SXT_rank == 1 |
                          SXT_rank == 2 ~ SXT_real,
                        TRUE ~ SXT_prob),
  ADAPT_NIT = case_when(NIT_rank == 1 |
                          NIT_rank == 2 ~ NIT_real,
                        TRUE ~ NIT_prob),
  ADAPT_LVX = case_when(LVX_rank == 1 |
                          LVX_rank == 2 ~ LVX_real,
                        TRUE ~ LVX_prob),
  ADAPT_CIP = case_when(CIP_rank == 1 |
                          CIP_rank == 2 ~ CIP_real,
                        TRUE ~ CIP_prob)
)

#Split test result returns as separate data frame
ADAPT_AST_1 <- ABX_rank[,grep("ADAPT", colnames(ABX_rank))]

#Convert 0s and 1s to 'S' and 'R'
ADAPT_AST_1$ADAPT_SXT[ADAPT_AST_1$ADAPT_SXT==0] <- "S"
ADAPT_AST_1$ADAPT_SXT[ADAPT_AST_1$ADAPT_SXT==1] <- "R"
ADAPT_AST_1$ADAPT_NIT[ADAPT_AST_1$ADAPT_NIT==0] <- "S"
ADAPT_AST_1$ADAPT_NIT[ADAPT_AST_1$ADAPT_NIT==1] <- "R"
ADAPT_AST_1$ADAPT_LVX[ADAPT_AST_1$ADAPT_LVX==0] <- "S"
ADAPT_AST_1$ADAPT_LVX[ADAPT_AST_1$ADAPT_LVX==1] <- "R"
ADAPT_AST_1$ADAPT_CIP[ADAPT_AST_1$ADAPT_CIP==0] <- "S"
ADAPT_AST_1$ADAPT_CIP[ADAPT_AST_1$ADAPT_CIP==1] <- "R"


view(ADAPT_AST_1)

write.csv(ADAPT_AST_1, "/Users/alexhoward/Documents/Projects/PhD/ADAPT-AST/ADAPT_AST_1.csv")

