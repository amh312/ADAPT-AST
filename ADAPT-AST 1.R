
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
view(ABX_rank)
write.csv(ADAPT_AST_1, "/Users/alexhoward/Documents/Projects/PhD/ADAPT-AST/ADAPT_AST_1.csv", row.names=FALSE)

