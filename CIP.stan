//CIP random intercept model

data {
  
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
}
