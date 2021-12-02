from os import X_OK
from ab_tests import Bayesian


A_impressions, A_conversions = 17000, 30
B_impressions, B_conversions = 17000, 50

bayesian = Bayesian(A_impressions, A_conversions, B_impressions, B_conversions)

print(bayesian.get_uplift())

bayesian.plot(x_stop=0.015)
