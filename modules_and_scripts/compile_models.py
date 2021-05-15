from pystan import StanModel
import pickle
print('This script compiles the stan C++ models for MCMC sampling from the posterior. Please be patient')
#"Fast Hierarchical Model"
model_circle_hierarchical_fast_code="""
functions {
real likelihood_lpdf(data vector PTRM, data vector NRM, data int N, real R, real x_c, real y_c)
  {

    return -N/2. *log(sum(square(sqrt(square(PTRM-x_c)+square(NRM-y_c)) - sqrt(square(R)))));
  }
}
data {
  int I; // Number of observations for all specimens
  int M; // Number of specimens
  vector[I] PTRM; //PTRM data all specimens
  vector[I] NRM; //NRM data all specimens
  int N[M]; // Group sizes
  vector[M] PTRMmax;
  vector[M] B_labs;
  vector[M] dmax;
  vector[M] centroid;
  real priorstd;

}
parameters {
  real<lower=0,upper=250> int_site;
  real<lower=0> sd_site;
  vector[M] k; //Scale of possible k values based on distance to data.
  //R cannot be > dist_to_edge or we have the issue that we could fit our data to a circle on the other side
  vector<lower=0>[M] dist_to_edge; //distance to edge of circle where data are located
  real c; //"Slope Angle" for linear regression
  //real sq; //"Square" coefficient for linear regression
  vector[M] int_real;


}
transformed parameters {
  vector<lower=0>[M] phi= pi()/2-atan(int_real .*PTRMmax ./ B_labs);
  vector[M] R = 1 ./ k;
  vector[M] x_c=cos(phi) .* (dist_to_edge+R);
  vector[M] y_c=sin(phi) .* (dist_to_edge+R);

}

model {
  int pos;
  pos = 1;
  dist_to_edge ~ uniform(0, 2. *centroid); //dont expect data to be >>1 ever really.
  k ~ uniform(to_vector(N) ./ -dmax,to_vector(N) ./dmax); //stops fitting wrong side of circle
  sd_site ~ normal(0,priorstd);
  int_real ~ normal(int_site + k*c,sd_site);
  //target+=-log(sd_site);
  for (m in 1:M) {

    target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
    pos = pos + N[m];
    }
}
"""
#"Slow" Hierarchical Model
model_circle_hierarchical_slow_code="""
functions {
real likelihood_lpdf(data vector PTRM, data vector NRM, data int N, real R, real x_c, real y_c)
{

  return -N/2. *log(sum(square(sqrt(square(PTRM-x_c)+square(NRM-y_c)) - sqrt(square(R)))));
}
}
data {
int I; // Number of observations for all specimens
int M; // Number of specimens
vector[I] PTRM; //PTRM data all specimens
vector[I] NRM; //NRM data all specimens
int N[M]; // Group sizes
vector[M] PTRMmax;
vector[M] B_labs;
vector[M] dmax;
vector[M] centroid;
real priorstd;

}
parameters {
real<lower=0,upper=250> int_site;
vector[M] k; //Scale of possible k values based on distance to data.
//R cannot be > dist_to_edge or we have the issue that we could fit our data to a circle on the other side
vector<lower=0>[M] dist_to_edge; //distance to edge of circle where data are located
//real sq; //"Square" coefficient for linear regression
vector[M] int_raw;
real<lower=0> sd_site;
real c;

}
transformed parameters {


vector[M] R = 1 ./ k;
vector[M] int_real= int_site + k*c + int_raw*sd_site;
vector<lower=0>[M] phi= pi()/2-atan(int_real .*PTRMmax ./ B_labs);

vector[M] x_c=cos(phi) .* (dist_to_edge+R);
vector[M] y_c=sin(phi) .* (dist_to_edge+R);

}
model {
int pos;
pos = 1;
dist_to_edge ~ uniform(0, 2. *centroid); //dont expect data to be >>1 ever really.
k ~ uniform(to_vector(N) ./ -dmax,to_vector(N) ./dmax); //stops fitting wrong side of circle
sd_site ~ normal(0,priorstd);
//target+=-log(sd_site);
int_raw ~ std_normal();
//target+=-log(sd_site);
for (m in 1:M) {

  target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
  pos = pos + N[m];
  }
}
"""
#Compile models
model_circle_hierarchical_fast=StanModel(model_code=model_circle_hierarchical_fast_code)
model_circle_hierarchical_slow=StanModel(model_code=model_circle_hierarchical_slow_code)

#save models to pickle

with open('model_circle_slow.pkl', 'wb') as f:
    pickle.dump(model_circle_hierarchical_slow, f)
with open('model_circle_fast.pkl','wb') as f:
    pickle.dump(model_circle_hierarchical_fast, f)

print('Models successfully compiled!')
