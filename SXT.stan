//SXT random intercept model

data {
  
  // Data dimensions
  int<lower=1> n_tr_SXT; // Training sample size
  int<lower=1> x_n_SXT; // Number of beta coefficients
  int<lower=1> pt_n_SXT; // Number of categories
  int<lower=1> n_te_SXT; // Testing sample size
  
  //Standard data inputs
  matrix[n_tr_SXT,x_n_SXT] tr_x_SXT; // Training coefficient matrix
  int<lower=0, upper=1> tr_y_SXT[n_tr_SXT]; // Training outcome measure
  matrix[n_te_SXT,x_n_SXT] te_x_SXT; //Testing coefficient matrix

  // Random effects data inputs
  int<lower=1> tr_pt_SXT[n_tr_SXT]; // Training data category labels
  int<lower=1> te_pt_SXT[n_te_SXT]; // Testing data category labels
    
  // Standard priors
  real a1_m_SXT;
  real<lower=0> a1_s_SXT;
  real B_m_SXT;
  real<lower=0> B_s_SXT;
  
  // Random effects priors
  real<lower=0> ar_m_SXT;
  real<lower=0> ar_sa_SXT; // Prior shape: sigsq_alpha1
  real<lower=0> ar_sb_SXT; // Prior rate: sigsq_alpha1
}



parameters {
  
  //Standard parameters
  real a1_SXT; //overall intercept
  vector[pt_n_SXT] ar_SXT; //intercept adjustment according to group
  vector[x_n_SXT] B_SXT; //coefficient slope
  
  //Random effects parameters
  real<lower=0> ar_s_SXT;   // Variance of intercept adjustments
}



model {
  
  // Likelihood model
  tr_y_SXT ~ bernoulli_logit(a1_SXT + tr_x_SXT * B_SXT);
  
  // Coefficient priors
  a1_SXT ~ normal(a1_m_SXT , a1_s_SXT);
  ar_SXT ~ normal(ar_m_SXT , sqrt(ar_s_SXT));
  to_vector(B_SXT) ~ normal(B_m_SXT , B_s_SXT);
  
  // Priors on variances for random effects
  ar_s_SXT ~ gamma(ar_sa_SXT, ar_sb_SXT);;
}



generated quantities {
  
  // Generation of logit predictions on test dataset
  vector[n_te_SXT] SXT_y_pr;
  for (n in 1:n_te_SXT) {
    real log_prob = a1_SXT + te_x_SXT[n,] * B_SXT;
    
  // Inverse logit to derive probabilities
      SXT_y_pr[n] = inv_logit(log_prob);
    }
  }


