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
library("cmdstanr")
library("posterior")
library("rethinking")
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

#SXT

SXT_mcmc <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_SXT; // Training sample size
  int<lower=1> x_n_SXT; // Number of beta coefficients
  int<lower=1> age_group_n_SXT; // Number of categories
  int<lower=1> n_te_SXT; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_SXT,x_n_SXT] tr_x_SXT; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_SXT[n_tr_SXT]; // Training outcome measure
  matrix[n_te_SXT,x_n_SXT] te_x_SXT; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_SXT[n_tr_SXT]; // Training data category labels
  int<lower=1> te_age_group_SXT[n_te_SXT]; // Testing data category labels
    
  // Standard priors
  real a1_m_SXT;
  real<lower=0> a1_s_SXT;
  real B_m_SXT;
  real<lower=0> B_s_SXT;
  
  // Random effects priors
  real<lower=0> ar_m_SXT;
  real<lower=0> ar_sa_SXT; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_SXT; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_SXT;
  real<lower=0> br_sa_SXT; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_SXT; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_SXT; //overall intercept
  vector[age_group_n_SXT] ar_SXT; //intercept adjustment according to group
  matrix[age_group_n_SXT, x_n_SXT] br_SXT; //slope adjustment according to group
  vector[x_n_SXT] B_SXT; //coefficient slope
  
  //Random effects parameters
  real<lower=0> ar_s_SXT;   // Variance of intercept adjustments
  real<lower=0> br_s_SXT;   // Variance of intercept adjustments
}



model {
  // Likelihood model
  for (i in 1:n_tr_SXT) {
    real log_prob = a1_SXT + ar_SXT[tr_age_group_SXT[i]]
      + dot_product(tr_x_SXT[i,], (br_SXT[tr_age_group_SXT[i],] + B_SXT'));
    tr_y_SXT[i] ~ bernoulli_logit(log_prob);
  }

  
  // Coefficient priors
  a1_SXT ~ normal(a1_m_SXT, a1_s_SXT);
  ar_SXT ~ normal(ar_m_SXT, sqrt(ar_s_SXT));
  for (j in 1:age_group_n_SXT) {
    br_SXT[j,] ~ normal(0, sqrt(br_s_SXT));
  }
  to_vector(B_SXT) ~ normal(B_m_SXT, B_s_SXT);

  // Priors on variances for random effects
  ar_s_SXT ~ gamma(ar_sa_SXT, ar_sb_SXT);
  br_s_SXT ~ gamma(br_sa_SXT, br_sb_SXT);
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_SXT] SXT_y_pr;
  for (n in 1:n_te_SXT) {
    real log_prob = a1_SXT + ar_SXT[te_age_group_SXT[n]]
      + dot_product(te_x_SXT[n,], (br_SXT[te_age_group_SXT[n],] + B_SXT'));
    
    // Inverse logit to derive probabilities
    SXT_y_pr[n] = inv_logit(log_prob);
  }
}                          ")

mod_SXT_mcmc <- cmdstan_model(SXT_mcmc)

fit_SXT_mcmc <- mod_SXT_mcmc$sample(data = SXT_stan_amr_uti, seed=1)



#NIT

NIT_mcmc <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_NIT; // Training sample size
  int<lower=1> x_n_NIT; // Number of beta coefficients
  int<lower=1> age_group_n_NIT; // Number of categories
  int<lower=1> n_te_NIT; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_NIT,x_n_NIT] tr_x_NIT; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_NIT[n_tr_NIT]; // Training outcome measure
  matrix[n_te_NIT,x_n_NIT] te_x_NIT; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_NIT[n_tr_NIT]; // Training data category labels
  int<lower=1> te_age_group_NIT[n_te_NIT]; // Testing data category labels
    
  // Standard priors
  real a1_m_NIT;
  real<lower=0> a1_s_NIT;
  real B_m_NIT;
  real<lower=0> B_s_NIT;
  
  // Random effects priors
  real<lower=0> ar_m_NIT;
  real<lower=0> ar_sa_NIT; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_NIT; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_NIT;
  real<lower=0> br_sa_NIT; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_NIT; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_NIT; //overall intercept
  vector[age_group_n_NIT] ar_NIT; //intercept adjustment according to group
  matrix[age_group_n_NIT, x_n_NIT] br_NIT; //slope adjustment according to group
  vector[x_n_NIT] B_NIT; //coefficient slope
  
  //Random effects parameters
  real<lower=0> ar_s_NIT;   // Variance of intercept adjustments
  real<lower=0> br_s_NIT;   // Variance of intercept adjustments
}



model {
  // Likelihood model
  for (i in 1:n_tr_NIT) {
    real log_prob = a1_NIT + ar_NIT[tr_age_group_NIT[i]]
      + dot_product(tr_x_NIT[i,], (br_NIT[tr_age_group_NIT[i],] + B_NIT'));
    tr_y_NIT[i] ~ bernoulli_logit(log_prob);
  }

  
  // Coefficient priors
  a1_NIT ~ normal(a1_m_NIT, a1_s_NIT);
  ar_NIT ~ normal(ar_m_NIT, sqrt(ar_s_NIT));
  for (j in 1:age_group_n_NIT) {
    br_NIT[j,] ~ normal(0, sqrt(br_s_NIT));
  }
  to_vector(B_NIT) ~ normal(B_m_NIT, B_s_NIT);

  // Priors on variances for random effects
  ar_s_NIT ~ gamma(ar_sa_NIT, ar_sb_NIT);
  br_s_NIT ~ gamma(br_sa_NIT, br_sb_NIT);
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_NIT] NIT_y_pr;
  for (n in 1:n_te_NIT) {
    real log_prob = a1_NIT + ar_NIT[te_age_group_NIT[n]]
      + dot_product(te_x_NIT[n,], (br_NIT[te_age_group_NIT[n],] + B_NIT'));
    
    // Inverse logit to derive probabilities
    NIT_y_pr[n] = inv_logit(log_prob);
  }
} ")

mod_NIT_mcmc <- cmdstan_model(NIT_mcmc)

fit_NIT_mcmc <- mod_NIT_mcmc$sample(data = NIT_stan_amr_uti, seed=2)





#LVX

LVX_mcmc <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_LVX; // Training sample size
  int<lower=1> x_n_LVX; // Number of beta coefficients
  int<lower=1> age_group_n_LVX; // Number of categories
  int<lower=1> n_te_LVX; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_LVX,x_n_LVX] tr_x_LVX; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_LVX[n_tr_LVX]; // Training outcome measure
  matrix[n_te_LVX,x_n_LVX] te_x_LVX; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_LVX[n_tr_LVX]; // Training data category labels
  int<lower=1> te_age_group_LVX[n_te_LVX]; // Testing data category labels
    
  // Standard priors
  real a1_m_LVX;
  real<lower=0> a1_s_LVX;
  real B_m_LVX;
  real<lower=0> B_s_LVX;
  
  // Random effects priors
  real<lower=0> ar_m_LVX;
  real<lower=0> ar_sa_LVX; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_LVX; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_LVX;
  real<lower=0> br_sa_LVX; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_LVX; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_LVX; //overall intercept
  vector[age_group_n_LVX] ar_LVX; //intercept adjustment according to group
  matrix[age_group_n_LVX, x_n_LVX] br_LVX; //slope adjustment according to group
  vector[x_n_LVX] B_LVX; //coefficient slope
  
  //Random effects parameters
  real<lower=0> ar_s_LVX;   // Variance of intercept adjustments
  real<lower=0> br_s_LVX;   // Variance of intercept adjustments
}



model {
  // Likelihood model
  for (i in 1:n_tr_LVX) {
    real log_prob = a1_LVX + ar_LVX[tr_age_group_LVX[i]]
      + dot_product(tr_x_LVX[i,], (br_LVX[tr_age_group_LVX[i],] + B_LVX'));
    tr_y_LVX[i] ~ bernoulli_logit(log_prob);
  }

  
  // Coefficient priors
  a1_LVX ~ normal(a1_m_LVX, a1_s_LVX);
  ar_LVX ~ normal(ar_m_LVX, sqrt(ar_s_LVX));
  for (j in 1:age_group_n_LVX) {
    br_LVX[j,] ~ normal(0, sqrt(br_s_LVX));
  }
  to_vector(B_LVX) ~ normal(B_m_LVX, B_s_LVX);

  // Priors on variances for random effects
  ar_s_LVX ~ gamma(ar_sa_LVX, ar_sb_LVX);
  br_s_LVX ~ gamma(br_sa_LVX, br_sb_LVX);
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_LVX] LVX_y_pr;
  for (n in 1:n_te_LVX) {
    real log_prob = a1_LVX + ar_LVX[te_age_group_LVX[n]]
      + dot_product(te_x_LVX[n,], (br_LVX[te_age_group_LVX[n],] + B_LVX'));
    
    // Inverse logit to derive probabilities
    LVX_y_pr[n] = inv_logit(log_prob);
  }
}")

mod_LVX_mcmc <- cmdstan_model(LVX_mcmc)

fit_LVX_mcmc <- mod_LVX_mcmc$sample(data = LVX_stan_amr_uti, seed=3)





#CIP

CIP_mcmc <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_CIP; // Training sample size
  int<lower=1> x_n_CIP; // Number of beta coefficients
  int<lower=1> age_group_n_CIP; // Number of categories
  int<lower=1> n_te_CIP; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_CIP,x_n_CIP] tr_x_CIP; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_CIP[n_tr_CIP]; // Training outcome measure
  matrix[n_te_CIP,x_n_CIP] te_x_CIP; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_CIP[n_tr_CIP]; // Training data category labels
  int<lower=1> te_age_group_CIP[n_te_CIP]; // Testing data category labels
    
  // Standard priors
  real a1_m_CIP;
  real<lower=0> a1_s_CIP;
  real B_m_CIP;
  real<lower=0> B_s_CIP;
  
  // Random effects priors
  real<lower=0> ar_m_CIP;
  real<lower=0> ar_sa_CIP; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_CIP; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_CIP;
  real<lower=0> br_sa_CIP; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_CIP; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_CIP; //overall intercept
  vector[age_group_n_CIP] ar_CIP; //intercept adjustment according to group
  matrix[age_group_n_CIP, x_n_CIP] br_CIP; //slope adjustment according to group
  vector[x_n_CIP] B_CIP; //coefficient slope
  
  //Random effects parameters
  real<lower=0> ar_s_CIP;   // Variance of intercept adjustments
  real<lower=0> br_s_CIP;   // Variance of intercept adjustments
}



model {
  // Likelihood model
  for (i in 1:n_tr_CIP) {
    real log_prob = a1_CIP + ar_CIP[tr_age_group_CIP[i]]
      + dot_product(tr_x_CIP[i,], (br_CIP[tr_age_group_CIP[i],] + B_CIP'));
    tr_y_CIP[i] ~ bernoulli_logit(log_prob);
  }

  
  // Coefficient priors
  a1_CIP ~ normal(a1_m_CIP, a1_s_CIP);
  ar_CIP ~ normal(ar_m_CIP, sqrt(ar_s_CIP));
  for (j in 1:age_group_n_CIP) {
    br_CIP[j,] ~ normal(0, sqrt(br_s_CIP));
  }
  to_vector(B_CIP) ~ normal(B_m_CIP, B_s_CIP);

  // Priors on variances for random effects
  ar_s_CIP ~ gamma(ar_sa_CIP, ar_sb_CIP);
  br_s_CIP ~ gamma(br_sa_CIP, br_sb_CIP);
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_CIP] CIP_y_pr;
  for (n in 1:n_te_CIP) {
    real log_prob = a1_CIP + ar_CIP[te_age_group_CIP[n]]
      + dot_product(te_x_CIP[n,], (br_CIP[te_age_group_CIP[n],] + B_CIP'));
    
    // Inverse logit to derive probabilities
    CIP_y_pr[n] = inv_logit(log_prob);
  }
} ")

mod_CIP_mcmc <- cmdstan_model(CIP_mcmc)

fit_CIP_mcmc <- mod_CIP_mcmc$sample(data = CIP_stan_amr_uti, seed=4)





# 4. TEST DATASET PREDICTIONS

#SXT

SXT_test <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_SXT; // Training sample size
  int<lower=1> x_n_SXT; // Number of beta coefficients
  int<lower=1> age_group_n_SXT; // Number of categories
  int<lower=1> n_te_SXT; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_SXT,x_n_SXT] tr_x_SXT; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_SXT[n_tr_SXT]; // Training outcome measure
  matrix[n_te_SXT,x_n_SXT] te_x_SXT; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_SXT[n_tr_SXT]; // Training data category labels
  int<lower=1> te_age_group_SXT[n_te_SXT]; // Testing data category labels
    
  // Standard priors
  real a1_m_SXT;
  real<lower=0> a1_s_SXT;
  real B_m_SXT;
  real<lower=0> B_s_SXT;
  
  // Random effects priors
  real<lower=0> ar_m_SXT;
  real<lower=0> ar_sa_SXT; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_SXT; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_SXT;
  real<lower=0> br_sa_SXT; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_SXT; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_SXT; //overall intercept
  vector[age_group_n_SXT] ar_SXT; //intercept adjustment according to group
  matrix[age_group_n_SXT, x_n_SXT] br_SXT; //slope adjustment according to group
  vector[x_n_SXT] B_SXT; //coefficient slope
  
  //Random effects parameters
  real<lower=1e-100> ar_s_SXT;   // SD of intercept adjustments
  real<lower=1e-100> br_s_SXT;   // SD of intercept adjustments
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_SXT] SXT_y_pr;
  for (n in 1:n_te_SXT) {
    real log_prob = a1_SXT + ar_SXT[te_age_group_SXT[n]]
      + dot_product(te_x_SXT[n,], (br_SXT[te_age_group_SXT[n],] + B_SXT'));
    
    // Inverse logit to derive probabilities
    SXT_y_pr[n] = inv_logit(log_prob);
  }
}                          ")

mod_SXT_test <- cmdstan_model(SXT_test)

fit_SXT_test <- mod_SXT_test$generate_quantities(fit_SXT_mcmc,
                                             data=SXT_stan_amr_uti,
                                             seed=1)

SXT_test_df <- as_draws_df(fit_SXT_test$draws())





#NIT

NIT_test <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_NIT; // Training sample size
  int<lower=1> x_n_NIT; // Number of beta coefficients
  int<lower=1> age_group_n_NIT; // Number of categories
  int<lower=1> n_te_NIT; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_NIT,x_n_NIT] tr_x_NIT; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_NIT[n_tr_NIT]; // Training outcome measure
  matrix[n_te_NIT,x_n_NIT] te_x_NIT; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_NIT[n_tr_NIT]; // Training data category labels
  int<lower=1> te_age_group_NIT[n_te_NIT]; // Testing data category labels
    
  // Standard priors
  real a1_m_NIT;
  real<lower=0> a1_s_NIT;
  real B_m_NIT;
  real<lower=0> B_s_NIT;
  
  // Random effects priors
  real<lower=0> ar_m_NIT;
  real<lower=0> ar_sa_NIT; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_NIT; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_NIT;
  real<lower=0> br_sa_NIT; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_NIT; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_NIT; //overall intercept
  vector[age_group_n_NIT] ar_NIT; //intercept adjustment according to group
  matrix[age_group_n_NIT, x_n_NIT] br_NIT; //slope adjustment according to group
  vector[x_n_NIT] B_NIT; //coefficient slope
  
  //Random effects parameters
  real<lower=1e-100> ar_s_NIT;   // SD of intercept adjustments
  real<lower=1e-100> br_s_NIT;   // SD of intercept adjustments
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_NIT] NIT_y_pr;
  for (n in 1:n_te_NIT) {
    real log_prob = a1_NIT + ar_NIT[te_age_group_NIT[n]]
      + dot_product(te_x_NIT[n,], (br_NIT[te_age_group_NIT[n],] + B_NIT'));
    
    // Inverse logit to derive probabilities
    NIT_y_pr[n] = inv_logit(log_prob);
  }
}                          ")

mod_NIT_test <- cmdstan_model(NIT_test)

fit_NIT_test <- mod_NIT_test$generate_quantities(fit_NIT_mcmc,
                                                 data=NIT_stan_amr_uti,
                                                 seed=2)

NIT_test_df <- as_draws_df(fit_NIT_test$draws())





#LVX

LVX_test <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_LVX; // Training sample size
  int<lower=1> x_n_LVX; // Number of beta coefficients
  int<lower=1> age_group_n_LVX; // Number of categories
  int<lower=1> n_te_LVX; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_LVX,x_n_LVX] tr_x_LVX; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_LVX[n_tr_LVX]; // Training outcome measure
  matrix[n_te_LVX,x_n_LVX] te_x_LVX; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_LVX[n_tr_LVX]; // Training data category labels
  int<lower=1> te_age_group_LVX[n_te_LVX]; // Testing data category labels
    
  // Standard priors
  real a1_m_LVX;
  real<lower=0> a1_s_LVX;
  real B_m_LVX;
  real<lower=0> B_s_LVX;
  
  // Random effects priors
  real<lower=0> ar_m_LVX;
  real<lower=0> ar_sa_LVX; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_LVX; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_LVX;
  real<lower=0> br_sa_LVX; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_LVX; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_LVX; //overall intercept
  vector[age_group_n_LVX] ar_LVX; //intercept adjustment according to group
  matrix[age_group_n_LVX, x_n_LVX] br_LVX; //slope adjustment according to group
  vector[x_n_LVX] B_LVX; //coefficient slope
  
  //Random effects parameters
  real<lower=1e-100> ar_s_LVX;   // SD of intercept adjustments
  real<lower=1e-100> br_s_LVX;   // SD of intercept adjustments
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_LVX] LVX_y_pr;
  for (n in 1:n_te_LVX) {
    real log_prob = a1_LVX + ar_LVX[te_age_group_LVX[n]]
      + dot_product(te_x_LVX[n,], (br_LVX[te_age_group_LVX[n],] + B_LVX'));
    
    // Inverse logit to derive probabilities
    LVX_y_pr[n] = inv_logit(log_prob);
  }
}                          ")

mod_LVX_test <- cmdstan_model(LVX_test)

fit_LVX_test <- mod_LVX_test$generate_quantities(fit_LVX_mcmc,
                                                 data=LVX_stan_amr_uti,
                                                 seed=3)

LVX_test_df <- as_draws_df(fit_LVX_test$draws())



#CIP

CIP_test <- write_stan_file("data {
  
  // Data dimensions
  int<lower=1> n_tr_CIP; // Training sample size
  int<lower=1> x_n_CIP; // Number of beta coefficients
  int<lower=1> age_group_n_CIP; // Number of categories
  int<lower=1> n_te_CIP; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_CIP,x_n_CIP] tr_x_CIP; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_CIP[n_tr_CIP]; // Training outcome measure
  matrix[n_te_CIP,x_n_CIP] te_x_CIP; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_age_group_CIP[n_tr_CIP]; // Training data category labels
  int<lower=1> te_age_group_CIP[n_te_CIP]; // Testing data category labels
    
  // Standard priors
  real a1_m_CIP;
  real<lower=0> a1_s_CIP;
  real B_m_CIP;
  real<lower=0> B_s_CIP;
  
  // Random effects priors
  real<lower=0> ar_m_CIP;
  real<lower=0> ar_sa_CIP; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_CIP; // Prior rate: sigsq_alpha1
  real<lower=0> br_m_CIP;
  real<lower=0> br_sa_CIP; // Prior shape: sigsq_alpha1
  real<lower=0> br_sb_CIP; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_CIP; //overall intercept
  vector[age_group_n_CIP] ar_CIP; //intercept adjustment according to group
  matrix[age_group_n_CIP, x_n_CIP] br_CIP; //slope adjustment according to group
  vector[x_n_CIP] B_CIP; //coefficient slope
  
  //Random effects parameters
  real<lower=1e-100> ar_s_CIP;   // SD of intercept adjustments
  real<lower=1e-100> br_s_CIP;   // SD of intercept adjustments
}

generated quantities {
  // Generation of logit predictions on test dataset
  vector[n_te_CIP] CIP_y_pr;
  for (n in 1:n_te_CIP) {
    real log_prob = a1_CIP + ar_CIP[te_age_group_CIP[n]]
      + dot_product(te_x_CIP[n,], (br_CIP[te_age_group_CIP[n],] + B_CIP'));
    
    // Inverse logit to derive probabilities
    CIP_y_pr[n] = inv_logit(log_prob);
  }
}                          ")

mod_CIP_test <- cmdstan_model(CIP_test)

fit_CIP_test <- mod_CIP_test$generate_quantities(fit_CIP_mcmc,
                                                 data=CIP_stan_amr_uti,
                                                 seed=4)

CIP_test_df <- as_draws_df(fit_CIP_test$draws())






# 5. PREDICTION ASSESSMENTS

#SXT
df_SXT_model <- as.data.frame( SXT_test_df )
df_SXT_model <- as.data.frame( list(
  SXT_mean = apply(df_SXT_model , 2 , mean)))
df_SXT_model <- cbind(
  as.data.frame( list( SXT_actual = te_amr_uti$SXT ) ), 
  SXT_pred = df_SXT_model[ grep('SXT_y_pr' , rownames(df_SXT_model)) , ] )
LL_SXT <- LogLoss(df_SXT_model$SXT_pred ,
        df_SXT_model$SXT_actual )
SXT_pred <- prediction( df_SXT_model$SXT_pred , df_SXT_model$SXT_actual )
SXT_perf <- performance( SXT_pred , "tpr" , "fpr" )
AUC_SXT <- AUC( df_SXT_model$SXT_pred , df_SXT_model$SXT_actual )


#NIT
df_NIT_model <- as.data.frame( NIT_test_df )
df_NIT_model <- as.data.frame( list(
  NIT_mean = apply(df_NIT_model , 2 , mean)))
df_NIT_model <- cbind(
  as.data.frame( list( NIT_actual = te_amr_uti$NIT ) ), 
  NIT_pred = df_NIT_model[ grep('NIT_y_pr' , rownames(df_NIT_model)) , ] )
LL_NIT <- LogLoss(df_NIT_model$NIT_pred ,
        df_NIT_model$NIT_actual )
NIT_pred <- prediction( df_NIT_model$NIT_pred , df_NIT_model$NIT_actual )
NIT_perf <- performance( NIT_pred , "tpr" , "fpr" )
AUC_NIT <- AUC( df_NIT_model$NIT_pred , df_NIT_model$NIT_actual )


#LVX
df_LVX_model <- as.data.frame( LVX_test_df )
df_LVX_model <- as.data.frame( list(
  LVX_mean = apply(df_LVX_model , 2 , mean)))
df_LVX_model <- cbind(
  as.data.frame( list( LVX_actual = te_amr_uti$LVX ) ), 
  LVX_pred = df_LVX_model[ grep('LVX_y_pr' , rownames(df_LVX_model)) , ] )
LL_LVX <- LogLoss(df_LVX_model$LVX_pred ,
        df_LVX_model$LVX_actual )
LVX_pred <- prediction( df_LVX_model$LVX_pred , df_LVX_model$LVX_actual )
LVX_perf <- performance( LVX_pred , "tpr" , "fpr" )
AUC_LVX <- AUC( df_LVX_model$LVX_pred , df_LVX_model$LVX_actual )


#CIP
df_CIP_model <- as.data.frame( CIP_test_df )
df_CIP_model <- as.data.frame( list(
  CIP_mean = apply(df_CIP_model , 2 , mean)))
df_CIP_model <- cbind(
  as.data.frame( list( CIP_actual = te_amr_uti$CIP ) ), 
  CIP_pred = df_CIP_model[ grep('CIP_y_pr' , rownames(df_CIP_model)) , ] )
LL_CIP <- LogLoss(df_CIP_model$CIP_pred ,
        df_CIP_model$CIP_actual )
CIP_pred <- prediction( df_CIP_model$CIP_pred , df_CIP_model$CIP_actual )
CIP_perf <- performance( CIP_pred , "tpr" , "fpr" )
AUC_CIP <- AUC( df_CIP_model$CIP_pred , df_CIP_model$CIP_actual )


#AUC plots
par(mfrow = c(2,2))
plot( SXT_perf , colorize=TRUE, main="Cotrimoxazole")
plot( NIT_perf , colorize=TRUE, main = "Nitrofurantoin" )
plot( LVX_perf , colorize=TRUE, main = "Levofloxacin" )
plot( CIP_perf , colorize=TRUE, main = "Ciprofloxacin" )

test_AUCs <- rbind( Cotrimoxazole = AUC_SXT,
                    Nitrofurantoin = AUC_NIT,
                    Levofloxacin = AUC_LVX,
                    Ciprofloxacin = AUC_CIP)

test_LLs <- rbind(Cotrimoxazole = LL_SXT,
                  Nitrofurantoin = LL_NIT,
                  Levofloxacin = LL_LVX,
                  Ciprofloxacin = LL_CIP)

test_perf <- cbind( AUC = test_AUCs,
                    LogLoss = test_LLs)

colnames(test_perf) = c("AUC", "Log loss")

view(test_perf)

cat(paste("\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n", "Modelling provided by ADAPT_AST v1.0",
          "\n","\n","\n","\n","\n","\n",
          "Posterior sampling completed (4 chains, 1,0000 iterations)",
          "\n","\n","\n","\n","\n","\n",
          "NEXT MODEL UPDATE DUE:", 
          Sys.Date()+30, "\n","\n",
          "\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n"))


