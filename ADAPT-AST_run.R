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

write_csv(te_amr_uti[3,], "AAST1_input.csv")

# 1. INPUT DATA

AAST1_input <- read.csv("AAST1_input.csv")


# 2. GENERATE PREDICTIONS

#SXT

AAST1_x_SXT <- as.matrix(model.matrix( SXT ~
                                         B1_SXT + # Predictor variable 1
                                         B2_SXT +
                                         B3_SXT + # Predictor variable 1
                                         B4_SXT +
                                         B5_SXT ,
                                       data = AAST1_input )) #testing matrix formation

AAST1_x_SXT <- AAST1_x_SXT[,2:ncol(AAST1_x_SXT), drop=F] # Intercept removal
AAST1_input <- as.data.frame(AAST1_input) # Convert back to data frame

AAST1_SXT_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_SXT = nrow(tr_amr_uti) , # Training sample size
  n_te_SXT = nrow(AAST1_input) , # Testing sample size
  x_n_SXT = ncol(tr_x_SXT) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_SXT = tr_x_SXT , # Predictor variable amr_uti (training)
  tr_y_SXT = tr_amr_uti$SXT , # Outcome variable amr_uti (training)
  te_x_SXT = AAST1_x_SXT , # Predictor variable amr_uti (testing)
  
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
  te_age_group_SXT = AAST1_input$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_SXT = 0 ,  #Random intercept prior mean
  ar_sa_SXT = 0.5 , ar_sb_SXT = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_SXT = 0 ,  #Random slope prior mean
  br_sa_SXT = 0.5 , br_sb_SXT = 0.05 #Random slope sd prior alpha and beta
  
)

SXT_gq <- write_stan_file("data {
  
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

mod_SXT_gq <- cmdstan_model(SXT_gq)

fit_SXT_gq <- mod_SXT_gq$generate_quantities(fit_SXT_mcmc,
                                             data=AAST1_SXT_stan_amr_uti,
                                             seed=1)

SXT_preds <- as_draws_df(fit_SXT_gq$draws())

SXT_HPDIs <- round(apply(SXT_preds,2,HPDI, prob=0.95), digits=3)

SXT_means <- round(apply(SXT_preds,2,mean, prob=0.95, ), digits=3)




#NIT

AAST1_x_NIT <- as.matrix(model.matrix( NIT ~
                                         B1_NIT + # Predictor variable 1
                                         B2_NIT +
                                         B3_NIT + # Predictor variable 1
                                         B4_NIT +
                                         B5_NIT ,
                                       data = AAST1_input )) #testing matrix formation

AAST1_x_NIT <- AAST1_x_NIT[,2:ncol(AAST1_x_NIT), drop=F] # Intercept removal
AAST1_input <- as.data.frame(AAST1_input) # Convert back to data frame

AAST1_NIT_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_NIT = nrow(tr_amr_uti) , # Training sample size
  n_te_NIT = nrow(AAST1_input) , # Testing sample size
  x_n_NIT = ncol(tr_x_NIT) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_NIT = tr_x_NIT , # Predictor variable amr_uti (training)
  tr_y_NIT = tr_amr_uti$NIT , # Outcome variable amr_uti (training)
  te_x_NIT = AAST1_x_NIT , # Predictor variable amr_uti (testing)
  
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
  age_group_n_NIT = length(unique(amr_uti$age_group)) , # Number of group_idegories
  tr_age_group_NIT = tr_amr_uti$age_group , # category amr_uti (training)
  te_age_group_NIT = AAST1_input$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_NIT = 0 ,  #Random intercept prior mean
  ar_sa_NIT = 0.5 , ar_sb_NIT = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_NIT = 0 ,  #Random slope prior mean
  br_sa_NIT = 0.5 , br_sb_NIT = 0.05 #Random slope sd prior alpha and beta
  
)

NIT_gq <- write_stan_file("data {
  
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

mod_NIT_gq <- cmdstan_model(NIT_gq)

fit_NIT_gq <- mod_NIT_gq$generate_quantities(fit_NIT_mcmc,
                                             data=AAST1_NIT_stan_amr_uti,
                                             seed=2)

NIT_preds <- as_draws_df(fit_NIT_gq$draws())

NIT_HPDIs <- round(apply(NIT_preds,2,HPDI, prob=0.95), digits=3)

NIT_means <- round(apply(NIT_preds,2,mean, prob=0.95, ), digits=3)





#LVX

AAST1_x_LVX <- as.matrix(model.matrix( LVX ~
                                         B1_LVX + # Predictor variable 1
                                         B2_LVX +
                                         B3_LVX + # Predictor variable 1
                                         B4_LVX +
                                         B5_LVX +
                                         B6_LVX,
                                       data = AAST1_input )) #testing matrix formation

AAST1_x_LVX <- AAST1_x_LVX[,2:ncol(AAST1_x_LVX), drop=F] # Intercept removal
AAST1_input <- as.data.frame(AAST1_input) # Convert back to data frame

AAST1_LVX_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_LVX = nrow(tr_amr_uti) , # Training sample size
  n_te_LVX = nrow(AAST1_input) , # Testing sample size
  x_n_LVX = ncol(tr_x_LVX) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_LVX = tr_x_LVX , # Predictor variable amr_uti (training)
  tr_y_LVX = tr_amr_uti$LVX , # Outcome variable amr_uti (training)
  te_x_LVX = AAST1_x_LVX , # Predictor variable amr_uti (testing)
  
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
  te_age_group_LVX = AAST1_input$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_LVX = 0 ,  #Random intercept prior mean
  ar_sa_LVX = 0.5 , ar_sb_LVX = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_LVX = 0 ,  #Random slope prior mean
  br_sa_LVX = 0.5 , br_sb_LVX = 0.05 #Random slope sd prior alpha and beta
  
)

LVX_gq <- write_stan_file("data {
  
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

mod_LVX_gq <- cmdstan_model(LVX_gq)

fit_LVX_gq <- mod_LVX_gq$generate_quantities(fit_LVX_mcmc,
                                             data=AAST1_LVX_stan_amr_uti,
                                             seed=3)

LVX_preds <- as_draws_df(fit_LVX_gq$draws())

LVX_HPDIs <- round(apply(LVX_preds,2,HPDI, prob=0.95), digits=3)

LVX_means <- round(apply(LVX_preds,2,mean, prob=0.95, ), digits=3)



#CIP

AAST1_x_CIP <- as.matrix(model.matrix( CIP ~
                                         B1_CIP + # Predictor variable 1
                                         B2_CIP +
                                         B3_CIP + # Predictor variable 1
                                         B4_CIP +
                                         B5_CIP +
                                         B6_CIP,
                                       data = AAST1_input )) #testing matrix formation

AAST1_x_CIP <- AAST1_x_CIP[,2:ncol(AAST1_x_CIP), drop=F] # Intercept removal
AAST1_input <- as.data.frame(AAST1_input) # Convert back to data frame

AAST1_CIP_stan_amr_uti <- list(
  
  # 4.1 Data dimensions
  n_tr_CIP = nrow(tr_amr_uti) , # Training sample size
  n_te_CIP = nrow(AAST1_input) , # Testing sample size
  x_n_CIP = ncol(tr_x_CIP) , # Number of beta coefficients
  
  # 4.2 Standard data inputs
  tr_x_CIP = tr_x_CIP , # Predictor variable amr_uti (training)
  tr_y_CIP = tr_amr_uti$CIP , # Outcome variable amr_uti (training)
  te_x_CIP = AAST1_x_CIP , # Predictor variable amr_uti (testing)
  
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
  te_age_group_CIP = AAST1_input$age_group , # category amr_uti (testing)
  
  # 4.5 Random intercept priors
  ar_m_CIP = 0 ,  #Random intercept prior mean
  ar_sa_CIP = 0.5 , ar_sb_CIP = 0.05 , #Random intercept sd prior alpha and beta
  
  # 4.6 Random slope priors
  br_m_CIP = 0 ,  #Random slope prior mean
  br_sa_CIP = 0.5 , br_sb_CIP = 0.05 #Random slope sd prior alpha and beta
  
)

CIP_gq <- write_stan_file("data {
  
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

mod_CIP_gq <- cmdstan_model(CIP_gq)

fit_CIP_gq <- mod_CIP_gq$generate_quantities(fit_CIP_mcmc,
                                             data=AAST1_CIP_stan_amr_uti,
                                             seed=4)

CIP_preds <- as_draws_df(fit_CIP_gq$draws())

CIP_HPDIs <- round(apply(CIP_preds,2,HPDI, prob=0.95), digits=3)

CIP_means <- round(apply(CIP_preds,2,mean, prob=0.95, ), digits=3)


# 3. PRESENT PROBABILITIES AND MAKE SELECTIONS


AAST1_HPDIs <- cbind(Cotrimoxazole=SXT_HPDIs[,1],
                     Nitrofurantoin=NIT_HPDIs[,1],
                     Levofloxacin=LVX_HPDIs[,1],
                     Ciprofloxacin=CIP_HPDIs[,1])

AAST1_means <- cbind(Cotrimoxazole=SXT_means[1],
                     Nitrofurantoin=NIT_means[1],
                     Levofloxacin=LVX_means[1],
                     Ciprofloxacin=CIP_means[1])

AAST1_output <- rbind(AAST1_means,
                      AAST1_HPDIs)

rownames(AAST1_output)[1] = "mean"

AAST1_unc <- abs(0.5-AAST1_output)
AAST1_unc <- AAST1_unc[2,]+AAST1_unc[3,]
AAST1_rank <- as.data.frame(rank(AAST1_unc))


mid <- barplot(AAST1_output[1,])
barplot(AAST1_output[1,], ylim=c(0,1), col=rainbow(10))
arrows(x0=mid, y0=AAST1_output[2,],
       x1=mid, y1=AAST1_output[3,],
       code=3, angle=90, length=0.1)


TEST_1 <- row.names(AAST1_rank)[which(AAST1_rank$`rank(AAST1_unc)`==1)]
TEST_2 <- row.names(AAST1_rank)[which(AAST1_rank$`rank(AAST1_unc)`==2)]


view(AAST1_output)

cat(paste("\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n",
          "\n","PLEASE TEST:","\n","\n",TEST_1,"\n","+","\n",TEST_2,
          "\n","\n","\n","\n","\n","\n",
          "Probability predictions provided by ADAPT-AST v1.0","\n","\n","\n","\n",
          "Variables used to make predictions:","\n","\n",
          "Previous antimicrobial resistance",
          "\n","Previous antimicrobial prescription(s)",
          "\n","Nursing home residency",
          "\n","Sampling location (inpatient or outpatient",
          "\n","\n","\n","\n","\n","\n","\n",
          "\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n","\n"))

