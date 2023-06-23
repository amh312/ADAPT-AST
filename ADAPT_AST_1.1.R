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
AAST1_group <- AAST1_input$age_group
ar_SXT <- subset(SXT_model_df, select = colnames(SXT_model_df) == paste("ar_SXT[",AAST1_group,"]", sep=""))
br_SXT <- subset(SXT_model_df, select = grepl(paste("br_SXT\\[", AAST1_group,",", sep = ""), colnames(SXT_model_df)))
B_SXT <- subset(SXT_model_df, select = grepl(paste("B_SXT", sep = ""), colnames(SXT_model_df)))
AAST1_x_SXT <- as.matrix(model.matrix( SXT ~ B1_SXT +
                                          B2_SXT +
                                          B3_SXT +
                                          B4_SXT +
                                          B5_SXT ,
                                          data = AAST1_input ))
AAST1_x_SXT <- AAST1_x_SXT[,2:ncol(AAST1_x_SXT)]
AAST1_input <- as.data.frame(AAST1_input)

# GENERATE PREDICTIONS
  
## SXT

log_prob <- SXT_model_df$a1_SXT + ar_SXT +
    rowSums(AAST1_x_SXT * (br_SXT + B_SXT))
SXT_y_pr <- 1 / (1 + exp(-log_prob))

HPDI(SXT_y_pr)
