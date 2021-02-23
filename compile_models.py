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
#squared fast model
model_circle_square_fast_code="""
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
  real sq; //"Square" coefficient for linear regression
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
  int_real ~ normal(int_site + k*c + square(k)*(sq),sd_site);
  //target+=-log(sd_site);
  for (m in 1:M) {

    target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
    pos = pos + N[m];
    }
}
"""
#squared slow model
model_circle_square_slow_code="""
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
real sq; //"Square" coefficient for linear regression
vector[M] int_raw;
real<lower=0> sd_site;
real c;

}
transformed parameters {


vector[M] R = 1 ./ k;
//matrix[1,1] R_ast = qr_R(to_matrix(k))[1:1, ]*sqrt(M-1);
//matrix[M,1] Q_ast = qr_Q(to_matrix(k))[ ,1:1]/sqrt(M-1);
//matrix[1,1] R_ast_inverse= inverse(R_ast);
vector[M] int_real= int_site + k*c + square(k)*sq + int_raw*sd_site;
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
#cubed fast model
model_circle_cube_fast_code="""
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
  real sq; //"Square" coefficient for linear regression
  real cu;
  vector[M] int_real;


}
transformed parameters {
  vector<lower=0>[M] phi= pi()/2-atan(int_real .*PTRMmax ./ B_labs);
  vector[M] R = 1 ./ k;
  vector[M] x_c=cos(phi) .* (dist_to_edge+R);
  vector[M] y_c=sin(phi) .* (dist_to_edge+R);
  vector[M] k_cube;
  for (m in 1:M) {
  k_cube[m]=pow(k[m],3);
  }

}

model {
  int pos;
  pos = 1;
  dist_to_edge ~ uniform(0, 2. *centroid); //dont expect data to be >>1 ever really.
  k ~ uniform(to_vector(N) ./ -dmax,to_vector(N) ./dmax); //stops fitting wrong side of circle
  sd_site ~ normal(0,priorstd);
  int_real ~ normal(int_site + k*c + square(k)*(sq) + k_cube*cu,sd_site);
  //target+=-log(sd_site);
  for (m in 1:M) {

    target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
    pos = pos + N[m];
    }
}
"""
#cubed slow model
model_circle_cube_slow_code="""
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
real sq; //"Square" coefficient for linear regression
vector[M] int_raw;
real<lower=0> sd_site;
real c;
real cu;

}
transformed parameters {
vector[M] R = 1 ./ k;
vector[M] k_cube;

vector[M] int_real= int_site + k*c + square(k)*sq + k_cube*cu + int_raw*sd_site;
vector<lower=0>[M] phi= pi()/2-atan(int_real .*PTRMmax ./ B_labs);

vector[M] x_c=cos(phi) .* (dist_to_edge+R);
vector[M] y_c=sin(phi) .* (dist_to_edge+R);
for (m in 1:M) {
k_cube[m]=pow(k[m],3);
}
}
model {
int pos;
pos = 1;
dist_to_edge ~ uniform(0, 2. *centroid); //dont expect data to be >>1 ever really.
k ~ uniform(to_vector(N) ./ -dmax,to_vector(N) ./dmax); //stops fitting wrong side of circle
sd_site ~ normal(0,priorstd);
int_raw ~ std_normal();
for (m in 1:M) {

  target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
  pos = pos + N[m];
  }
}
"""
#unpooled model
model_circle_code="""functions {
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
  vector[M] dmax;
  vector[M] centroid;

}
parameters {
  vector<lower=0>[M] dist_to_edge; //Scale of possible k values based on distance to data.
  //R cannot be > dist_to_edge or we have the issue that we could fit our data to a circle on the other side
  vector[M] k; //distance to edge of circle where data are located
  vector<lower=0,upper=pi()>[M] phi; //Angle to line joining circle center to origin
}
transformed parameters{
  //vector[M] k= k_scale ./ dist_to_edge;
  vector[M] R = 1 ./ k;
  vector[M] x_c=cos(phi) .* (dist_to_edge+R);
  vector[M] y_c=sin(phi) .* (dist_to_edge+R);
  vector[M] slope=1 ./tan(phi) ./PTRMmax;
}

model {
  int pos;
  pos = 1;
  k ~ uniform(to_vector(N) ./ -dmax,to_vector(N) ./dmax); //stops fitting wrong side of circle
  //k_scale ~ uniform(-0.75,1);
  phi ~ uniform(0,pi()); //Makes tangent function unique
  dist_to_edge ~ uniform(0,2*centroid);
  for (m in 1:M) {

    target+=likelihood_lpdf(segment(PTRM, pos, N[m])|segment(NRM, pos, N[m]),N[m],R[m],x_c[m],y_c[m]);
    pos = pos + N[m];
    }
}
"""

#Compile models
model_circle_hierarchical_fast=StanModel(model_code=model_circle_hierarchical_fast_code)
model_circle_hierarchical_slow=StanModel(model_code=model_circle_hierarchical_slow_code)
model_circle_square_fast=StanModel(model_code=model_circle_square_fast_code)
model_circle_square_slow=StanModel(model_code=model_circle_square_slow_code)
model_circle_cube_fast=StanModel(model_code=model_circle_cube_fast_code)
model_circle_cube_slow=StanModel(model_code=model_circle_cube_slow_code)
model_circle=StanModel(model_code=model_circle_code)

#save models to pickle

with open('model_circle_slow.pkl', 'wb') as f:
    pickle.dump(model_circle_hierarchical_slow, f)
with open('model_circle_fast.pkl','wb') as f:
    pickle.dump(model_circle_hierarchical_fast, f)
with open('model_circle_unpooled.pkl','wb') as f:
    pickle.dump(model_circle, f)
with open('model_circle_square_fast.pkl','wb') as f:
    pickle.dump(model_circle_square_fast, f)
with open('model_circle_square_slow.pkl','wb') as f:
    pickle.dump(model_circle_square_slow, f)
with open('model_circle_cube_fast.pkl','wb') as f:
    pickle.dump(model_circle_cube_fast, f)
with open('model_circle_cube_slow.pkl','wb') as f:
    pickle.dump(model_circle_cube_fast, f)

print('Models successfully compiled!')
quit()
