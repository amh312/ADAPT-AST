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
library("rethinking")
setwd("/Users/alexhoward/Documents/Projects/PhD/ADAPT-AST")
load("~/Documents/Projects/PhD/ADAPT-AST/ABX.RData")



AAST1_input <- te_amr_uti[1,]

# GENERATE PREDICTIONS
  
## SXT
  
log_prob <- sapply(AAST_input, function(x)
  SXT_model_df$a1_SXT + SXT_model_df$ar_SXT[AAST1_input$te_age_group_SXT[n]] +
    sum(te_x_SXT[n,] * (SXT_model_df$br_SXT[AAST1_input$te_age_group_SXT[n],] +
                          SXT_model_df$B_SXT)))
  



view(log_prob)

SXT_y_pr[n] <- exp(log_prob) / (1 + exp(log_prob))

SXT_model_df <- as.data.frame(SXT_model)

view(AAST1_input)

ncol(SXT_model_df)
