//NIT random intercept model

data {
  
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
}
