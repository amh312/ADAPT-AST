//SXT random intercept model

data {
  
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
}
