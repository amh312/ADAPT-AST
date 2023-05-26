//LVX random intercept model

data {
  
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
}
